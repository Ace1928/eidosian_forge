from __future__ import absolute_import
import os
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
def _build_from_model(self, vectors, labels, num_features):
    try:
        from annoy import AnnoyIndex
    except ImportError:
        raise _NOANNOY
    index = AnnoyIndex(num_features, metric='angular')
    for vector_num, vector in enumerate(vectors):
        index.add_item(vector_num, vector)
    index.build(self.num_trees)
    self.index = index
    self.labels = labels