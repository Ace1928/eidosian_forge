import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _str_subex(self, subex):
    s = '%s' % subex
    if isinstance(subex, DrtConcatenation) and subex.consequent is None:
        return s[1:-1]
    return s