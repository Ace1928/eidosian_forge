import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __rv_french(self, word, vowels):
    """
        Return the region RV that is used by the French stemmer.

        If the word begins with two vowels, RV is the region after
        the third letter. Otherwise, it is the region after the first
        vowel not at the beginning of the word, or the end of the word
        if these positions cannot be found. (Exceptionally, u'par',
        u'col' or u'tap' at the beginning of a word is also taken to
        define RV as the region to their right.)

        :param word: The French word whose region RV is determined.
        :type word: str or unicode
        :param vowels: The French vowels that are used to determine
                       the region RV.
        :type vowels: unicode
        :return: the region RV for the respective French word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of
               the subclass FrenchStemmer. It is not to be invoked directly!

        """
    rv = ''
    if len(word) >= 2:
        if word.startswith(('par', 'col', 'tap')) or (word[0] in vowels and word[1] in vowels):
            rv = word[3:]
        else:
            for i in range(1, len(word)):
                if word[i] in vowels:
                    rv = word[i + 1:]
                    break
    return rv