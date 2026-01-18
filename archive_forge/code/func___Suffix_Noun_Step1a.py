import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Noun_Step1a(self, token):
    for suffix in self.__suffix_noun_step1a:
        if token.endswith(suffix):
            if suffix in self.__conjugation_suffix_noun_1 and len(token) >= 4:
                token = token[:-1]
                self.suffix_noun_step1a_success = True
                break
            if suffix in self.__conjugation_suffix_noun_2 and len(token) >= 5:
                token = token[:-2]
                self.suffix_noun_step1a_success = True
                break
            if suffix in self.__conjugation_suffix_noun_3 and len(token) >= 6:
                token = token[:-3]
                self.suffix_noun_step1a_success = True
                break
    return token