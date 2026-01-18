import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __Prefix_Step3b_Noun(self, token):
    for prefix in self.__prefix_step3b_noun:
        if token.startswith(prefix):
            if len(token) > 3:
                if prefix == 'ب':
                    token = token[len(prefix):]
                    self.prefix_step3b_noun_success = True
                    break
                if prefix in self.__prepositions2:
                    token = prefix_replace(token, prefix, prefix[1])
                    self.prefix_step3b_noun_success = True
                    break
            if prefix in self.__prepositions1 and len(token) > 4:
                token = token[len(prefix):]
                self.prefix_step3b_noun_success = True
                break
    return token