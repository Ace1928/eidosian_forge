import logging
import unittest
from gensim import matutils
from scipy.sparse import csr_matrix
import numpy as np
import math
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import ldamodel
from gensim.test.utils import datapath, common_dictionary, common_corpus
class TestHellinger(unittest.TestCase):

    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(common_corpus, id2word=common_dictionary, num_topics=2, passes=100)

    def test_inputs(self):
        vec_1 = []
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)
        vec_1 = np.array([])
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)
        vec_1 = csr_matrix([])
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

    def test_distributions(self):
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.484060507634
        self.assertAlmostEqual(expected, result)
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1), (8, 0.1), (10, 0.8), (9, 0.1)]
        result = matutils.hellinger(vec_1, vec_2)
        result_symmetric = matutils.hellinger(vec_2, vec_1)
        expected = 0.856921568786
        self.assertAlmostEqual(expected, result)
        self.assertAlmostEqual(expected, result_symmetric)
        vec_1 = np.array([[1, 0.3], [0, 0.4], [2, 0.3]])
        vec_2 = csr_matrix([[1, 0.4], [0, 0.2], [2, 0.2]])
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.160618030536
        self.assertAlmostEqual(expected, result)
        vec_1 = np.array([0.6, 0.1, 0.1, 0.2])
        vec_2 = [0.2, 0.2, 0.1, 0.5]
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.309742984153
        self.assertAlmostEqual(expected, result)
        np.random.seed(0)
        model = self.class_(self.corpus, id2word=common_dictionary, num_topics=2, passes=100)
        lda_vec1 = model[[(1, 2), (2, 3)]]
        lda_vec2 = model[[(2, 2), (1, 3)]]
        result = matutils.hellinger(lda_vec1, lda_vec2)
        expected = 1.0406845281146034e-06
        self.assertAlmostEqual(expected, result)