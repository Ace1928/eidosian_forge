from __future__ import absolute_import
import pickle as _pickle
from smart_open import open
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
def _build_from_keyedvectors(self):
    """Build an NMSLIB index using word vectors from a KeyedVectors model."""
    self._build_from_model(self.model.get_normed_vectors(), self.model.index_to_key)