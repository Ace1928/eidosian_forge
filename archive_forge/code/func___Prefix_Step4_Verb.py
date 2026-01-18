import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step4_Verb(self, token):
    for prefix in self.__prefix_step4_verb:
        if token.startswith(prefix) and len(token) > 4:
            token = prefix_replace(token, prefix, 'است')
            self.is_verb = True
            self.is_noun = False
            break
    return token