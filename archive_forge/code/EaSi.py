from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Email messages and their labels
emails = [
    ("Buy cheap viagra now!", "spam"),
    ("Important meeting tomorrow at 10am", "ham"),
    ("Congratulations! You have won a prize!", "spam"),
    ("Hey, how are you doing?", "ham"),
    ("Get rich quick with this amazing opportunity!", "spam"),
]

# Separate the email messages and labels
messages = [email[0] for email in emails]
labels = [email[1] for email in emails]

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Test the classifier on new email messages
new_emails = [
    "You have been selected for a free gift!",
    "Please review the attached document",
    "Lose weight fast with this secret formula!",
]
X_new = vectorizer.transform(new_emails)
predictions = classifier.predict(X_new)

# Print the predictions
for email, prediction in zip(new_emails, predictions):
    print("Email:", email)
    print("Prediction:", prediction)
    print("---")
