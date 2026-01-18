import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _pad_random(m, new_rows, rand):
    """Pad a matrix with additional rows filled with random values."""
    _, columns = m.shape
    low, high = (-1.0 / columns, 1.0 / columns)
    suffix = rand.uniform(low, high, (new_rows, columns)).astype(REAL)
    return vstack([m, suffix])