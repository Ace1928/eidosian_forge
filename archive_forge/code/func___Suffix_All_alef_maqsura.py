import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Suffix_All_alef_maqsura(self, token):
    for suffix in self.__suffix_all_alef_maqsura:
        if token.endswith(suffix):
            token = suffix_replace(token, suffix, 'ي')
    return token