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
class TestWmdSimilarity(_TestSimilarityABC):

    def setUp(self):
        self.cls = similarities.WmdSimilarity
        self.w2v_model = Word2Vec(TEXTS, min_count=1).wv

    def factoryMethod(self):
        return self.cls(TEXTS, self.w2v_model)

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_full(self, num_best=None):
        index = self.cls(TEXTS, self.w2v_model)
        index.num_best = num_best
        query = TEXTS[0]
        sims = index[query]
        if num_best is not None:
            for i, sim in sims:
                self.assertTrue(numpy.alltrue(sim > 0.0))
        else:
            self.assertTrue(sims[0] == 1.0)
            self.assertTrue(numpy.alltrue(sims[1:] > 0.0))
            self.assertTrue(numpy.alltrue(sims[1:] < 1.0))

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_non_increasing(self):
        """ Check that similarities are non-increasing when `num_best` is not
        `None`."""
        index = self.cls(TEXTS, self.w2v_model, num_best=3)
        query = TEXTS[0]
        sims = index[query]
        sims2 = numpy.asarray(sims)[:, 1]
        cond = sum(numpy.diff(sims2) < 0) == len(sims2) - 1
        self.assertTrue(cond)

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_chunking(self):
        index = self.cls(TEXTS, self.w2v_model)
        query = TEXTS[:3]
        sims = index[query]
        for i in range(3):
            self.assertTrue(numpy.alltrue(sims[i, i] == 1.0))
        index.num_best = 3
        sims = index[query]
        for sims_temp in sims:
            for i, sim in sims_temp:
                self.assertTrue(numpy.alltrue(sim > 0.0))
                self.assertTrue(numpy.alltrue(sim <= 1.0))

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_iter(self):
        index = self.cls(TEXTS, self.w2v_model)
        for sims in index:
            self.assertTrue(numpy.alltrue(sims >= 0.0))
            self.assertTrue(numpy.alltrue(sims <= 1.0))

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_str(self):
        index = self.cls(TEXTS, self.w2v_model)
        self.assertTrue(str(index))