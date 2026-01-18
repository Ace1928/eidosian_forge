import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _init_post_load(self, hidden_output):
    num_vectors = len(self.wv.vectors)
    vocab_size = len(self.wv)
    vector_size = self.wv.vector_size
    assert num_vectors > 0, 'expected num_vectors to be initialized already'
    assert vocab_size > 0, 'expected vocab_size to be initialized already'
    self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)
    self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
    if self.hs:
        self.syn1 = hidden_output
    if self.negative:
        self.syn1neg = hidden_output
    self.layer1_size = vector_size