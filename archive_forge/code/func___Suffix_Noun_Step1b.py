import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Noun_Step1b(self, token):
    for suffix in self.__suffix_noun_step1b:
        if token.endswith(suffix) and len(token) > 5:
            token = token[:-1]
            self.suffixe_noun_step1b_success = True
            break
    return token