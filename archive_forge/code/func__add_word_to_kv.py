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
def _add_word_to_kv(kv, counts, word, weights, vocab_size):
    if kv.has_index_for(word):
        logger.warning("duplicate word '%s' in word2vec file, ignoring all but first", word)
        return
    word_id = kv.add_vector(word, weights)
    if counts is None:
        word_count = vocab_size - word_id
    elif word in counts:
        word_count = counts[word]
    else:
        logger.warning("vocabulary file is incomplete: '%s' is missing", word)
        word_count = None
    kv.set_vecattr(word, 'count', word_count)