import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
class TestUniformTermSimilarityIndex(unittest.TestCase):

    def setUp(self):
        self.documents = [[u'government', u'denied', u'holiday'], [u'holiday', u'slowing', u'hollingworth']]
        self.dictionary = Dictionary(self.documents)

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        index = UniformTermSimilarityIndex(self.dictionary)
        results = list(index.most_similar(u'holiday', topn=1))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(1, len(results))
        results = list(index.most_similar(u'holiday', topn=4))
        self.assertLess(1, len(results))
        self.assertGreaterEqual(4, len(results))
        index = UniformTermSimilarityIndex(self.dictionary)
        terms = [term for term, similarity in index.most_similar(u'holiday', topn=len(self.dictionary))]
        self.assertFalse(u'holiday' in terms)
        index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.2)
        similarities = numpy.array([similarity for term, similarity in index.most_similar(u'holiday', topn=len(self.dictionary))])
        self.assertTrue(numpy.all(similarities == 0.2))