import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestHessenberg:

    def test_simple(self):
        a = [[-149, -50, -154], [537, 180, 546], [-27, -9, -25]]
        h1 = [[-149.0, 42.2037, -156.3165], [-537.6783, 152.5511, -554.9272], [0, 0.0728, 2.4489]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)
        assert_array_almost_equal(h, h1, decimal=4)

    def test_simple_complex(self):
        a = [[-149, -50, -154], [537, 180j, 546], [-27j, -9, -25]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.conj().T @ a @ q, h)

    def test_simple2(self):
        a = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 6, 7, 2], [0, 2, 2, 3, 0, 3, 2], [0, 0, 2, 8, 0, 0, 2], [0, 3, 1, 2, 0, 1, 2], [0, 1, 2, 3, 0, 1, 0], [0, 0, 0, 0, 0, 1, 2]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)

    def test_simple3(self):
        a = np.eye(3)
        a[-1, 0] = 2
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)

    def test_random(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            h, q = hessenberg(a, calc_q=1)
            assert_array_almost_equal(q.T @ a @ q, h)

    def test_random_complex(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            h, q = hessenberg(a, calc_q=1)
            assert_array_almost_equal(q.conj().T @ a @ q, h)

    def test_check_finite(self):
        a = [[-149, -50, -154], [537, 180, 546], [-27, -9, -25]]
        h1 = [[-149.0, 42.2037, -156.3165], [-537.6783, 152.5511, -554.9272], [0, 0.0728, 2.4489]]
        h, q = hessenberg(a, calc_q=1, check_finite=False)
        assert_array_almost_equal(q.T @ a @ q, h)
        assert_array_almost_equal(h, h1, decimal=4)

    def test_2x2(self):
        a = [[2, 1], [7, 12]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q, np.eye(2))
        assert_array_almost_equal(h, a)
        b = [[2 - 7j, 1 + 2j], [7 + 3j, 12 - 2j]]
        h2, q2 = hessenberg(b, calc_q=1)
        assert_array_almost_equal(q2, np.eye(2))
        assert_array_almost_equal(h2, b)