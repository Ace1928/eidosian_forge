import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _least_check(self, valence, words_and_emoticons, i):
    if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon and (words_and_emoticons[i - 1].lower() == 'least'):
        if words_and_emoticons[i - 2].lower() != 'at' and words_and_emoticons[i - 2].lower() != 'very':
            valence = valence * self.constants.N_SCALAR
    elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon and (words_and_emoticons[i - 1].lower() == 'least'):
        valence = valence * self.constants.N_SCALAR
    return valence