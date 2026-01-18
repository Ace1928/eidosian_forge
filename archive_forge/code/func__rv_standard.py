import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def _rv_standard(self, word, vowels):
    """
        Return the standard interpretation of the string region RV.

        If the second letter is a consonant, RV is the region after the
        next following vowel. If the first two letters are vowels, RV is
        the region after the next following consonant. Otherwise, RV is
        the region after the third letter.

        :param word: The word whose region RV is determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the region RV.
        :type vowels: unicode
        :return: the region RV for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the respective stem method of
               the subclasses ItalianStemmer, PortugueseStemmer,
               RomanianStemmer, and SpanishStemmer. It is not to be
               invoked directly!

        """
    rv = ''
    if len(word) >= 2:
        if word[1] not in vowels:
            for i in range(2, len(word)):
                if word[i] in vowels:
                    rv = word[i + 1:]
                    break
        elif word[0] in vowels and word[1] in vowels:
            for i in range(2, len(word)):
                if word[i] not in vowels:
                    rv = word[i + 1:]
                    break
        else:
            rv = word[3:]
    return rv