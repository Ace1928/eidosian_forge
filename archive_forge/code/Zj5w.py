from googletrans import Translator, LANGUAGES

# Text to translate
text = "Bonjour, comment allez-vous?"

# Create a translator object
translator = Translator()

# Translate the text
try:
    # Detect the language of the input text
    detected_language = translator.detect(text).lang
    if detected_language in LANGUAGES.keys():
        translation = translator.translate(text, dest="en")
        print("Original Text:", text)
        print("Translated Text:", translation.text)
    else:
        print(f"Translation from {detected_language} to English is not supported.")
except Exception as e:
    print(f"An error occurred during translation: {e}")
