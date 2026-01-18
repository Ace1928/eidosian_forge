import re
from nltk.stem.api import StemmerI
def __getLastLetter(self, word):
    """Get the zero-based index of the last alphabetic character in this string"""
    last_letter = -1
    for position in range(len(word)):
        if word[position].isalpha():
            last_letter = position
        else:
            break
    return last_letter