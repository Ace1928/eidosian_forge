import argostranslate.package
import argostranslate.translate

# Text to translate
text = "Bonjour, comment allez-vous?"

# Ensure the latest packages are available and install necessary language packages
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()


# Install packages for detected language to English if not already installed
def install_language_package(from_code, to_code="en"):
    package_to_install = next(
        (
            pkg
            for pkg in available_packages
            if pkg.from_code == from_code and pkg.to_code == to_code
        ),
        None,
    )
    if package_to_install:
        argostranslate.package.install_from_path(package_to_install.download())
    else:
        print(f"No available package to translate from {from_code} to {to_code}")


# Load installed languages
installed_languages = argostranslate.translate.get_installed_languages()
language_codes = {lang.code: lang for lang in installed_languages}

# Assuming English is installed
to_lang = next((lang for lang in installed_languages if lang.code == "en"), None)

try:
    # Attempt to translate directly to English and infer the source language
    if to_lang:
        translation = to_lang.translate_from(text)
        print("Original Text:", text)
        print("Translated Text:", translation)
    else:
        print("English translation not supported.")
except Exception as e:
    print(f"An error occurred during translation: {e}")
