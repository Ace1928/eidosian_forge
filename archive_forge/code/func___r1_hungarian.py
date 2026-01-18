import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __r1_hungarian(self, word, vowels, digraphs):
    """
        Return the region R1 that is used by the Hungarian stemmer.

        If the word begins with a vowel, R1 is defined as the region
        after the first consonant or digraph (= two letters stand for
        one phoneme) in the word. If the word begins with a consonant,
        it is defined as the region after the first vowel in the word.
        If the word does not contain both a vowel and consonant, R1
        is the null region at the end of the word.

        :param word: The Hungarian word whose region R1 is determined.
        :type word: str or unicode
        :param vowels: The Hungarian vowels that are used to determine
                       the region R1.
        :type vowels: unicode
        :param digraphs: The digraphs that are used to determine the
                         region R1.
        :type digraphs: tuple
        :return: the region R1 for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               HungarianStemmer. It is not to be invoked directly!

        """
    r1 = ''
    if word[0] in vowels:
        for digraph in digraphs:
            if digraph in word[1:]:
                r1 = word[word.index(digraph[-1]) + 1:]
                return r1
        for i in range(1, len(word)):
            if word[i] not in vowels:
                r1 = word[i + 1:]
                break
    else:
        for i in range(1, len(word)):
            if word[i] in vowels:
                r1 = word[i + 1:]
                break
    return r1