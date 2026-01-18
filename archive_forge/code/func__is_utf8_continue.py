import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _is_utf8_continue(b):
    return b & _MB_MASK == _MB_START