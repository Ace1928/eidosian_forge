from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _ninf_array(shape):
    res = np.empty(shape, np.float64)
    res.fill(-np.inf)
    return res