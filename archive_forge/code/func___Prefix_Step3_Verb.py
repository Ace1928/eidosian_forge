import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step3_Verb(self, token):
    for prefix in self.__prefix_step3_verb:
        if token.startswith(prefix) and len(token) > 4:
            token = prefix_replace(token, prefix, prefix[1])
            break
    return token