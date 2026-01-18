import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __normalize_pre(self, token):
    """
        :param token: string
        :return: normalized token type string
        """
    token = self.__vocalization.sub('', token)
    token = self.__kasheeda.sub('', token)
    token = self.__arabic_punctuation_marks.sub('', token)
    return token