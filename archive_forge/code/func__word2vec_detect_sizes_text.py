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
def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:
            break
        if vector_size:
            continue
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return (vocab_size, vector_size)