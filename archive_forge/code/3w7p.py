import argostranslate.package
import argostranslate.translate
import os
from langdetect import detect
import logging

logging.basicConfig(level=logging.INFO)

# Text to translate
text = "Bonjour, comment allez-vous?"


# Function to detect language robustly
def robust_detect_language(text: str) -> str:
    try:
        detected_language = detect(text)
        logging.info(f"Detected Language: {detected_language}")
        return detected_language
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return None


detected_language = robust_detect_language(text)


# Update and load translation packages
def update_and_load_packages():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    installed_languages = argostranslate.translate.get_installed_languages()
    language_codes = {lang.code: lang for lang in installed_languages}
    return available_packages, installed_languages, language_codes


available_packages, installed_languages, language_codes = update_and_load_packages()


# Function to download and verify translation packages
def manage_translation_package(
    detected_language_code: str, target_language_code: str = "en"
) -> bool:
    logging.info(
        f"Checking available packages for translation from {detected_language_code} to {target_language_code}"
    )
    for package in available_packages:
        if (
            package.from_code == detected_language_code
            and package.to_code == target_language_code
        ):
            model_path = f"{package.from_code}_to_{package.to_code}.argosmodel"
            if not os.path.exists(model_path):
                try:
                    package.download()
                    logging.info("Download successful, verifying package...")
                    if os.path.exists(model_path):
                        logging.info("Package verified successfully.")
                        return True
                    else:
                        logging.error("Package verification failed.")
                        return False
                except Exception as e:
                    logging.error(f"Failed to download package: {e}")
                    return False
            else:
                logging.info("Package already exists.")
                return True
    logging.warning("No suitable package found for download.")
    return False


# Enhanced language detection and translation
def translate_text(text: str, installed_languages: list):
    source_lang = robust_detect_language(text)
    if source_lang not in language_codes:
        if manage_translation_package(source_lang):
            _, installed_languages, _ = update_and_load_packages()
        else:
            logging.error(
                f"No available translation package from {source_lang} to English."
            )
            return

    to_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    if to_lang:
        try:
            translation = language_codes[source_lang].translate_to(text, to_lang)
            logging.info(f"Original Text: {text}")
            logging.info(f"Translated Text: {translation}")
        except Exception as e:
            logging.error(f"An error occurred during translation: {e}")
    else:
        logging.error(
            "Translation to English not supported or source language not detected."
        )


translate_text(text, installed_languages)
