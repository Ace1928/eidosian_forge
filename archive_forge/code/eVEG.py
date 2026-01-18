import nltk
from nltk.corpus import stopwords
from nltk.stem import

#WordNetLemmatizer
Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#Text document
document = "The quick brown foxes are jumping over the lazy dogs. They are having a great time playing in the park."
Tokenize the document
tokens = nltk.word_tokenize(document)
#Lowercase the tokens
tokens = [token.lower() for token in tokens]
#Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words]
#Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]
Print the preprocessed tokens
print("Preprocessed Tokens:", tokens)