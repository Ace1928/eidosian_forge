import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
class TestAlgorithm_2_2:

    def test_randn_inv(self):
        np.random.seed(1234)
        n = 20
        nsamples = 100
        for i in range(nsamples):
            t = np.random.randint(1, 4)
            n = np.random.randint(10, 41)
            A = scipy.linalg.inv(np.random.randn(n, n))
            g, ind = _algorithm_2_2(A, A.T, t)