import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
    """Arrange any special handling for the gensim.utils.SaveLoad protocol"""
    ignore = set(ignore).union(['buckets_word', 'vectors'])
    return super(FastTextKeyedVectors, self)._save_specials(fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)