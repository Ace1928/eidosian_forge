import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Verb_Step2c(self, token):
    for suffix in self.__suffix_verb_step2c:
        if token.endswith(suffix):
            if suffix == 'تمو' and len(token) >= 6:
                token = token[:-3]
                break
            if suffix == 'و' and len(token) >= 4:
                token = token[:-1]
                break
    return token