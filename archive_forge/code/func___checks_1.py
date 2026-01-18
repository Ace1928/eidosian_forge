import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __checks_1(self, token):
    for prefix in self.__checks1:
        if token.startswith(prefix):
            if prefix in self.__articles_3len and len(token) > 4:
                self.is_noun = True
                self.is_verb = False
                self.is_defined = True
                break
            if prefix in self.__articles_2len and len(token) > 3:
                self.is_noun = True
                self.is_verb = False
                self.is_defined = True
                break