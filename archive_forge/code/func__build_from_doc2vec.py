from __future__ import absolute_import
import pickle as _pickle
from smart_open import open
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
def _build_from_doc2vec(self):
    """Build an NMSLIB index using document vectors from a Doc2Vec model."""
    docvecs = self.model.dv
    labels = docvecs.index_to_key
    self._build_from_model(docvecs.get_normed_vectors(), labels)