import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _never_check(self, valence, words_and_emoticons, start_i, i):
    if start_i == 0:
        if self.constants.negated([words_and_emoticons[i - 1]]):
            valence = valence * self.constants.N_SCALAR
    if start_i == 1:
        if words_and_emoticons[i - 2] == 'never' and (words_and_emoticons[i - 1] == 'so' or words_and_emoticons[i - 1] == 'this'):
            valence = valence * 1.5
        elif self.constants.negated([words_and_emoticons[i - (start_i + 1)]]):
            valence = valence * self.constants.N_SCALAR
    if start_i == 2:
        if words_and_emoticons[i - 3] == 'never' and (words_and_emoticons[i - 2] == 'so' or words_and_emoticons[i - 2] == 'this') or (words_and_emoticons[i - 1] == 'so' or words_and_emoticons[i - 1] == 'this'):
            valence = valence * 1.25
        elif self.constants.negated([words_and_emoticons[i - (start_i + 1)]]):
            valence = valence * self.constants.N_SCALAR
    return valence