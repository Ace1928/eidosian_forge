import logging
import unittest
from gensim import matutils
from scipy.sparse import csr_matrix
import numpy as np
import math
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import ldamodel
from gensim.test.utils import datapath, common_dictionary, common_corpus
class TestJaccard(unittest.TestCase):

    def test_inputs(self):
        vec_1 = []
        vec_2 = []
        self.assertRaises(ZeroDivisionError, matutils.jaccard, vec_1, vec_2)

    def test_distributions(self):
        vec_1 = [(2, 1), (3, 4), (4, 1), (5, 1), (1, 1), (7, 2)]
        vec_2 = [(1, 1), (3, 8), (4, 1)]
        result = matutils.jaccard(vec_2, vec_1)
        expected = 1 - 0.3
        self.assertAlmostEqual(expected, result)
        vec_1 = np.array([[1, 3], [0, 4], [2, 3]])
        vec_2 = csr_matrix([[1, 4], [0, 2], [2, 2]])
        result = matutils.jaccard(vec_1, vec_2)
        expected = 1 - 0.388888888889
        self.assertAlmostEqual(expected, result)
        vec_1 = np.array([6, 1, 2, 3])
        vec_2 = [4, 3, 2, 5]
        result = matutils.jaccard(vec_1, vec_2)
        expected = 1 - 0.333333333333
        self.assertAlmostEqual(expected, result)