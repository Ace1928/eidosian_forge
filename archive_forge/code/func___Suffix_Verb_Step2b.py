import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_Verb_Step2b(self, token):
    for suffix in self.__suffix_verb_step2b:
        if token.endswith(suffix) and len(token) >= 5:
            token = token[:-2]
            self.suffix_verb_step2b_success = True
            break
    return token