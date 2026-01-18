import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step2a(self, token):
    for prefix in self.__prefix_step2a:
        if token.startswith(prefix) and len(token) > 5:
            token = token[len(prefix):]
            self.prefix_step2a_success = True
            break
    return token