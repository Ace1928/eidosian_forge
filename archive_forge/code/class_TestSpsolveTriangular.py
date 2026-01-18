import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
class TestSpsolveTriangular:

    def setup_method(self):
        use_solver(useUmfpack=False)

    def test_zero_diagonal(self):
        n = 5
        rng = np.random.default_rng(43876432987)
        A = rng.standard_normal((n, n))
        b = np.arange(n)
        A = scipy.sparse.tril(A, k=0, format='csr')
        x = spsolve_triangular(A, b, unit_diagonal=True, lower=True)
        A.setdiag(1)
        assert_allclose(A.dot(x), b)
        A = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0])
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'CSR matrix format is')
            spsolve_triangular(A, b, unit_diagonal=True)

    def test_singular(self):
        n = 5
        A = csr_matrix((n, n))
        b = np.arange(n)
        for lower in (True, False):
            assert_raises(scipy.linalg.LinAlgError, spsolve_triangular, A, b, lower=lower)

    @sup_sparse_efficiency
    def test_bad_shape(self):
        A = np.zeros((3, 4))
        b = ones((4, 1))
        assert_raises(ValueError, spsolve_triangular, A, b)
        A2 = csr_matrix(eye(3))
        b2 = array([1.0, 2.0])
        assert_raises(ValueError, spsolve_triangular, A2, b2)

    @sup_sparse_efficiency
    def test_input_types(self):
        A = array([[1.0, 0.0], [1.0, 2.0]])
        b = array([[2.0, 0.0], [2.0, 2.0]])
        for matrix_type in (array, csc_matrix, csr_matrix):
            x = spsolve_triangular(matrix_type(A), b, lower=True)
            assert_array_almost_equal(A.dot(x), b)

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    @sup_sparse_efficiency
    def test_random(self):

        def random_triangle_matrix(n, lower=True):
            A = scipy.sparse.random(n, n, density=0.1, format='coo')
            if lower:
                A = scipy.sparse.tril(A)
            else:
                A = scipy.sparse.triu(A)
            A = A.tocsr(copy=False)
            for i in range(n):
                A[i, i] = np.random.rand() + 1
            return A
        np.random.seed(1234)
        for lower in (True, False):
            for n in (10, 10 ** 2, 10 ** 3):
                A = random_triangle_matrix(n, lower=lower)
                for m in (1, 10):
                    for b in (np.random.rand(n, m), np.random.randint(-9, 9, (n, m)), np.random.randint(-9, 9, (n, m)) + np.random.randint(-9, 9, (n, m)) * 1j):
                        x = spsolve_triangular(A, b, lower=lower)
                        assert_array_almost_equal(A.dot(x), b)
                        x = spsolve_triangular(A, b, lower=lower, unit_diagonal=True)
                        A.setdiag(1)
                        assert_array_almost_equal(A.dot(x), b)