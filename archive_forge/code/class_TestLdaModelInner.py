import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
class TestLdaModelInner(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState()
        self.num_runs = 100
        self.num_topics = 100

    def test_log_sum_exp(self):
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input = rs.uniform(-1000, 1000, size=(self.num_topics, 1))
                known_good = logsumexp(input)
                test_values = matutils.logsumexp(input)
                msg = 'logsumexp failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def test_mean_absolute_difference(self):
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input1 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                input2 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                known_good = mean_absolute_difference(input1, input2)
                test_values = matutils.mean_absolute_difference(input1, input2)
                msg = 'mean_absolute_difference failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def test_dirichlet_expectation(self):
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input_1d = rs.uniform(0.01, 10000, size=(self.num_topics,))
                known_good = dirichlet_expectation(input_1d)
                test_values = matutils.dirichlet_expectation(input_1d)
                msg = 'dirichlet_expectation_1d failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)
                input_2d = rs.uniform(0.01, 10000, size=(1, self.num_topics))
                known_good = dirichlet_expectation(input_2d)
                test_values = matutils.dirichlet_expectation(input_2d)
                msg = 'dirichlet_expectation_2d failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)