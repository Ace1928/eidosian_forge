import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
class TestVsNumpyNorm:
    _sparse_types = (scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, scipy.sparse.dia_matrix, scipy.sparse.dok_matrix, scipy.sparse.lil_matrix)
    _test_matrices = ((np.arange(9) - 4).reshape((3, 3)), [[1, 2, 3], [-1, 1, 4]], [[1, 0, 3], [-1, 1, 4j]])

    def test_sparse_matrix_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                assert_allclose(spnorm(S), npnorm(M))
                assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
                assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
                assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
                assert_allclose(spnorm(S, 1), npnorm(M, 1))
                assert_allclose(spnorm(S, -1), npnorm(M, -1))

    def test_sparse_matrix_norms_with_axis(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in (None, (0, 1), (1, 0)):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in ('fro', np.inf, -np.inf, 1, -1):
                        assert_allclose(spnorm(S, ord, axis=axis), npnorm(M, ord, axis=axis))
                for axis in ((-2, -1), (-1, -2), (1, -2)):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    assert_allclose(spnorm(S, 'f', axis=axis), npnorm(M, 'f', axis=axis))
                    assert_allclose(spnorm(S, 'fro', axis=axis), npnorm(M, 'fro', axis=axis))

    def test_sparse_vector_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in (0, 1, -1, -2, (0,), (1,), (-1,), (-2,)):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in (None, 2, np.inf, -np.inf, 1, 0.5, 0.42):
                        assert_allclose(spnorm(S, ord, axis=axis), npnorm(M, ord, axis=axis))