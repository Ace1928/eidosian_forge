import argostranslate.package
import argostranslate.translate
import os
from langdetect import detect

# Text to translate
text = "Bonjour, comment allez-vous?"

# Detect the language
detected_language = detect(text)

# Print the detected language
print("Detected Language:", detected_language)

# Ensure the latest packages are available and install necessary language packages
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Load installed languages
installed_languages = argostranslate.translate.get_installed_languages()
language_codes = {lang.code: lang for lang in installed_languages}


# Function to download missing packages for detected language to English translation
def download_missing_package(detected_language_code: str) -> bool:
    for package in available_packages:
        if package.from_code == detected_language_code and package.to_code == "en":
            model_path = f"{package.from_code}_to_{package.to_code}.argosmodel"
            if not os.path.exists(model_path):
                print(
                    f"Downloading package for {package.from_code} to {package.to_code}"
                )
                package.download()
                return True
    return False


# Detect the language of the input text
def detect_language(text: str, installed_languages: list) -> object:
    detected_lang_code = detect(text)
    for lang in installed_languages:
        if lang.code == detected_lang_code:
            return lang
    return None


source_lang = detect_language(text, installed_languages)

# Assuming English is installed
to_lang = next((lang for lang in installed_languages if lang.code == "en"), None)

if not source_lang:
    if download_missing_package(detected_language):
        # Reload installed languages after downloading the package
        installed_languages = argostranslate.translate.get_installed_languages()
        source_lang = detect_language(text, installed_languages)
    else:
        print(f"No available translation package from {detected_language} to English.")

try:
    # Attempt to translate directly to English and infer the source language
    if source_lang and to_lang:
        translation = source_lang.translate_to(text, to_lang)
        print("Original Text:", text)
        print("Translated Text:", translation)
    else:
        print("Translation to English not supported or source language not detected.")
except Exception as e:
    print(f"An error occurred during translation: {e}")
