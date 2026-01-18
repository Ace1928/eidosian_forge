import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _pad_ones(m, new_len):
    """Pad array with additional entries filled with ones."""
    if len(m) > new_len:
        raise ValueError('the new number of rows %i must be greater than old %i' % (new_len, len(m)))
    new_arr = np.ones(new_len, dtype=REAL)
    new_arr[:len(m)] = m
    return new_arr