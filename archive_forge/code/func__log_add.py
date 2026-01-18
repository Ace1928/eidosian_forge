from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _log_add(*values):
    """
    Adds the logged values, returning the logarithm of the addition.
    """
    x = max(values)
    if x > -np.inf:
        sum_diffs = 0
        for value in values:
            sum_diffs += 2 ** (value - x)
        return x + np.log2(sum_diffs)
    else:
        return x