import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __replace_accented(self, word):
    """
        Replaces all accented letters on a word with their non-accented
        counterparts.

        :param word: A spanish word, with or without accents
        :type word: str or unicode
        :return: a word with the accented letters (á, é, í, ó, ú) replaced with
                 their non-accented counterparts (a, e, i, o, u)
        :rtype: str or unicode
        """
    return word.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')