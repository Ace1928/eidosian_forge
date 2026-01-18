import logging
import unittest
from gensim import matutils
from scipy.sparse import csr_matrix
import numpy as np
import math
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import ldamodel
from gensim.test.utils import datapath, common_dictionary, common_corpus
class TestKL(unittest.TestCase):

    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(common_corpus, id2word=common_dictionary, num_topics=2, passes=100)

    def test_inputs(self):
        vec_1 = []
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)
        vec_1 = np.array([])
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)
        vec_1 = csr_matrix([])
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

    def test_distributions(self):
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.kullback_leibler(vec_2, vec_1, 8)
        expected = 0.55451775
        self.assertAlmostEqual(expected, result, places=5)
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.kullback_leibler(vec_1, vec_2, 8)
        self.assertTrue(math.isinf(result))
        vec_1 = np.array([[1, 0.3], [0, 0.4], [2, 0.3]])
        vec_2 = csr_matrix([[1, 0.4], [0, 0.2], [2, 0.2]])
        result = matutils.kullback_leibler(vec_1, vec_2, 3)
        expected = 0.0894502
        self.assertAlmostEqual(expected, result, places=5)
        vec_1 = np.array([0.6, 0.1, 0.1, 0.2])
        vec_2 = [0.2, 0.2, 0.1, 0.5]
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.40659450877
        self.assertAlmostEqual(expected, result, places=5)
        np.random.seed(0)
        model = self.class_(self.corpus, id2word=common_dictionary, num_topics=2, passes=100)
        lda_vec1 = model[[(1, 2), (2, 3)]]
        lda_vec2 = model[[(2, 2), (1, 3)]]
        result = matutils.kullback_leibler(lda_vec1, lda_vec2)
        expected = 4.283407e-12
        self.assertAlmostEqual(expected, result, places=5)