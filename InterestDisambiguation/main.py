from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt


data = "interest.acl94.txt"

# STEP 1 : Pré-traitement des données
# Open the file
with open(data) as file:
    lines = file.readlines()

# Split the lines by $$ and remove brackets
splitted_lines = [line.split("$$")[0].replace("[", "").replace("]", "").replace("=", "").replace("MGMNP", "") for line in lines]

# Split all lines by words
splitted_lines = [line.split() for line in splitted_lines if line]

# Map words in a object with word and type
for i in range(len(splitted_lines)):
    for j in range(len(splitted_lines[i])):
        splitted_lines[i][j] = {"word": splitted_lines[i][j].split("/")[0], "type": splitted_lines[i][j].split("/")[1]}

# Step 2 : Caractéristiques des données
# 2.1 - Retirer les stopwords
with open("stoplist-english.txt") as file2:
    stopwords = file2.readlines()
    stopwords = [stopword.replace("\n", "") for stopword in stopwords]

# Retirer les ponctuations & les stopwords
for i in range(len(splitted_lines)):
    splitted_lines[i] = [word_obj for word_obj in splitted_lines[i] if word_obj["type"] != ":"] # Filtre les ponctuations de type :
    splitted_lines[i] = [word_obj for word_obj in splitted_lines[i] if word_obj["word"] != word_obj["type"]] # Filtrer les ponctuations générales
    splitted_lines[i] = [word_obj for word_obj in splitted_lines[i] if word_obj["word"] not in stopwords] # Retirer les stopwords

# Prepare data for stemming and keep sense
for i in range(len(splitted_lines)):
    interest_word = None
    interest_word_index = None
    for j in range(len(splitted_lines[i])):
        if splitted_lines[i][j]["word"].startswith("interest") and not splitted_lines[i][j]["word"].startswith("*") and splitted_lines[i][j]["word"] != "interest":
            interest_word = splitted_lines[i][j]["word"]
            splitted_lines[i][j]["word"] = interest_word.split("_")[0]
            interest_word_index = j
    class_no = interest_word.split("_")[1]
    splitted_lines[i] = {"wordlist": splitted_lines[i], "class": class_no, "word_index": interest_word_index}

# Stemming the word data
data = splitted_lines
porter = PorterStemmer()
for i in range(len(data)):
    for j in range(len(data[i]["wordlist"])):
        data[i]["wordlist"][j]["word"] = porter.stem(data[i]["wordlist"][j]["word"])

# Step 3 : Prepare data for training
# Fenetre contexte grandeur = 1
# exemple : [mot1, mot_class1, mot2, mot_class2, ...; classe = class_no]

nb_accuracies = []
dt_accuracies = []
rf_accuracies = []
svm_accuracies = []
mlp_accuracies = []
for context in range(1, 6):
    features_list = []
    labels_list = []
    context_size = context
    for example in data:
        example_array = []
        for i in range(-context_size, context_size + 1):
            if i == 0: continue
            index = 0 if example["word_index"] + i < 0 else example["word_index"] + i
            index = len(example["wordlist"]) - 1 if index >= len(example["wordlist"]) - 1 else index
            example_array.append(example["wordlist"][index]["word"])
            example_array.append(example["wordlist"][index]["type"])
        features_list.append(example_array)
        labels_list.append(example["class"])

    # Tfidf Vectorize the textual data to numerical
    vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    features = vectorizer.fit_transform(features_list)

    # 3.2 - Split the dataset in train and test
    split_index = round(features.shape[0] * 0.80)

    X_train = features[:split_index]
    X_test = features[split_index:]
    y_train = labels_list[:split_index]
    y_test = labels_list[split_index:]

    # Step 4 : Train the models
    nb = MultinomialNB()                 # Naive Bayes
    dt = DecisionTreeClassifier()   # Decision Tree
    rf = RandomForestClassifier()        # Random forests
    svm = SVC()                      # SVM
    mlp = MLPClassifier()                # MLP

    # Fit models to the training data
    nb.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    # Prediction on test data
    nb_predictions = nb.predict(X_test)
    dt_predictions = dt.predict(X_test)
    rf_predictions = rf.predict(X_test)
    svm_predictions = svm.predict(X_test)
    mlp_predictions = mlp.predict(X_test)

    # Accuracy of each classifier
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)

    # Save accuracy for context size training
    nb_accuracies.append(nb_accuracy)
    dt_accuracies.append(dt_accuracy)
    rf_accuracies.append(rf_accuracy)
    svm_accuracies.append(svm_accuracy)
    mlp_accuracies.append(mlp_accuracy)

    print("Results for context size : " + str(context_size))
    print("Accuracy of Naive Bayes classifier:", nb_accuracy)
    print("Accuracy of Decision Tree classifier:", dt_accuracy)
    print("Accuracy of Random Forest classifier:", rf_accuracy)
    print("Accuracy of SVM classifier:", svm_accuracy)
    print("Accuracy of MLP classifier:", mlp_accuracy)
    print("")

# Plot the result on accuracy
plt.plot(range(1, 6), nb_accuracies)
plt.title("Taux de succès naive bayes par grandeur de fenêtre de contexte")
plt.xlabel("Grandeur de fenêtre de contexte")
plt.ylabel("Taux de succès")
plt.show()

plt.plot(range(1, 6), dt_accuracies)
plt.title("Taux de succès arbre de décision par grandeur de fenêtre de contexte")
plt.xlabel("Grandeur de fenêtre de contexte")
plt.ylabel("Taux de succès")
plt.show()

plt.plot(range(1, 6), rf_accuracies)
plt.title("Taux de succès forêt aléatoire par grandeur de fenêtre de contexte")
plt.xlabel("Grandeur de fenêtre de contexte")
plt.ylabel("Taux de succès")
plt.show()

plt.plot(range(1, 6), svm_accuracies)
plt.title("Taux de succès SVM par grandeur de fenêtre de contexte")
plt.xlabel("Grandeur de fenêtre de contexte")
plt.ylabel("Taux de succès")
plt.show()

plt.plot(range(1, 6), mlp_accuracies)
plt.title("Taux de succès MLP par grandeur de fenêtre de contexte")
plt.xlabel("Grandeur de fenêtre de contexte")
plt.ylabel("Taux de succès")
plt.show()

plt.plot(range(1, 6), nb_accuracies, label="Taux de succès naive bayes")
plt.plot(range(1, 6), dt_accuracies, label="Taux de succès arbre de decision")
plt.plot(range(1, 6), rf_accuracies, label="Taux de succès foret aleatoire")
plt.plot(range(1, 6), svm_accuracies, label="Taux de succès SVM")
plt.plot(range(1, 6), mlp_accuracies, label="Taux de succès MLP")
plt.legend()
plt.show()


# Test : Different hidden neurons amount

# mlp_accuracies = []
# neurons_configs = [1, 10, 50, 100, 200, 400]
# for neurons_amount in neurons_configs:
#     features_list = []
#     labels_list = []
#     context_size = 5
#     for example in data:
#         example_array = []
#         for i in range(-context_size, context_size + 1):
#             if i == 0: continue
#             index = 0 if example["word_index"] + i < 0 else example["word_index"] + i
#             index = len(example["wordlist"]) - 1 if index >= len(example["wordlist"]) - 1 else index
#             example_array.append(example["wordlist"][index]["word"])
#             example_array.append(example["wordlist"][index]["type"])
#         features_list.append(example_array)
#         labels_list.append(example["class"])
#
#     # Tfidf Vectorize the textual data to numerical
#     vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
#     features = vectorizer.fit_transform(features_list)
#
#     # 3.2 - Split the dataset in train and test
#     split_index = round(features.shape[0] * 0.80)
#
#     X_train = features[:split_index]
#     X_test = features[split_index:]
#     y_train = labels_list[:split_index]
#     y_test = labels_list[split_index:]
#
#     mlp = MLPClassifier(hidden_layer_sizes=neurons_amount)
#     mlp.fit(X_train, y_train)
#     mlp_predictions = mlp.predict(X_test)
#     mlp_accuracy = accuracy_score(y_test, mlp_predictions)
#     mlp_accuracies.append(mlp_accuracy)
#
# # Plot results
# plt.plot(neurons_configs, mlp_accuracies)
# plt.title("Taux de succès MLP par nombre de neurones cachées")
# plt.xlabel("Nombre de neurones cachées")
# plt.ylabel("Taux de succès")
# plt.show()
