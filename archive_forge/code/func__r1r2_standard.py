import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def _r1r2_standard(self, word, vowels):
    """
        Return the standard interpretations of the string regions R1 and R2.

        R1 is the region after the first non-vowel following a vowel,
        or is the null region at the end of the word if there is no
        such non-vowel.

        R2 is the region after the first non-vowel following a vowel
        in R1, or is the null region at the end of the word if there
        is no such non-vowel.

        :param word: The word whose regions R1 and R2 are determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the regions R1 and R2.
        :type vowels: unicode
        :return: (r1,r2), the regions R1 and R2 for the respective word.
        :rtype: tuple
        :note: This helper method is invoked by the respective stem method of
               the subclasses DutchStemmer, FinnishStemmer,
               FrenchStemmer, GermanStemmer, ItalianStemmer,
               PortugueseStemmer, RomanianStemmer, and SpanishStemmer.
               It is not to be invoked directly!
        :note: A detailed description of how to define R1 and R2
               can be found at http://snowball.tartarus.org/texts/r1r2.html

        """
    r1 = ''
    r2 = ''
    for i in range(1, len(word)):
        if word[i] not in vowels and word[i - 1] in vowels:
            r1 = word[i + 1:]
            break
    for i in range(1, len(r1)):
        if r1[i] not in vowels and r1[i - 1] in vowels:
            r2 = r1[i + 1:]
            break
    return (r1, r2)