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
class TestCDF2RDF:

    def matmul(self, a, b):
        return np.einsum('...ij,...jk->...ik', a, b)

    def assert_eig_valid(self, w, v, x):
        assert_array_almost_equal(self.matmul(v, w), self.matmul(x, v))

    def test_single_array0x0real(self):
        X = np.empty((0, 0))
        w, v = (np.empty(0), np.empty((0, 0)))
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array2x2_real(self):
        X = np.array([[1, 2], [3, -1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array2x2_complex(self):
        X = np.array([[1, 2], [-2, 1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array3x3_real(self):
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array3x3_complex(self):
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_random_1d_stacked_arrays(self):
        for M in range(1, 7):
            np.random.seed(999999999)
            X = np.random.rand(100, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    def test_random_2d_stacked_arrays(self):
        for M in range(1, 7):
            X = np.random.rand(10, 10, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    def test_low_dimensionality_error(self):
        w, v = (np.empty(()), np.array((2,)))
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_not_square_error(self):
        w, v = (np.arange(3), np.arange(6).reshape(3, 2))
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_swapped_v_w_error(self):
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, v, w)

    def test_non_associated_error(self):
        w, v = (np.arange(3), np.arange(16).reshape(4, 4))
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_not_conjugate_pairs(self):
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6 + 1j]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, w, v)
        X = np.array([[[1, 2, 3], [1, 2, 3], [2, 5, 6 + 1j]], [[1, 2, 3], [1, 2, 3], [2, 5, 6 - 1j]]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, w, v)