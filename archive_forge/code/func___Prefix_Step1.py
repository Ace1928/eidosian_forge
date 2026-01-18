import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step1(self, token):
    for prefix in self.__prefix_step1:
        if token.startswith(prefix) and len(token) > 3:
            if prefix == 'أأ':
                token = prefix_replace(token, prefix, 'أ')
                break
            elif prefix == 'أآ':
                token = prefix_replace(token, prefix, 'آ')
                break
            elif prefix == 'أؤ':
                token = prefix_replace(token, prefix, 'ؤ')
                break
            elif prefix == 'أا':
                token = prefix_replace(token, prefix, 'ا')
                break
            elif prefix == 'أإ':
                token = prefix_replace(token, prefix, 'إ')
                break
    return token