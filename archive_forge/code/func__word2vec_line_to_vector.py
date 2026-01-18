import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(' ')
    word, weights = (parts[0], [datatype(x) for x in parts[1:]])
    return (word, weights)