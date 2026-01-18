import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class TestCSR(sparse_test_class()):

    @classmethod
    def spcreator(cls, *args, **kwargs):
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a csr_matrix is expensive')
            return csr_matrix(*args, **kwargs)
    math_dtypes = [np.bool_, np.int_, np.float64, np.complex128]

    def test_constructor1(self):
        b = array([[0, 4, 0], [3, 0, 0], [0, 2, 0]], 'd')
        bsp = csr_matrix(b)
        assert_array_almost_equal(bsp.data, [4, 3, 2])
        assert_array_equal(bsp.indices, [1, 0, 1])
        assert_array_equal(bsp.indptr, [0, 1, 2, 3])
        assert_equal(bsp.getnnz(), 3)
        assert_equal(bsp.format, 'csr')
        assert_array_equal(bsp.toarray(), b)

    def test_constructor2(self):
        b = zeros((6, 6), 'd')
        b[3, 4] = 5
        bsp = csr_matrix(b)
        assert_array_almost_equal(bsp.data, [5])
        assert_array_equal(bsp.indices, [4])
        assert_array_equal(bsp.indptr, [0, 0, 0, 0, 1, 1, 1])
        assert_array_almost_equal(bsp.toarray(), b)

    def test_constructor3(self):
        b = array([[1, 0], [0, 2], [3, 0]], 'd')
        bsp = csr_matrix(b)
        assert_array_almost_equal(bsp.data, [1, 2, 3])
        assert_array_equal(bsp.indices, [0, 1, 0])
        assert_array_equal(bsp.indptr, [0, 1, 2, 3])
        assert_array_almost_equal(bsp.toarray(), b)

    def test_constructor4(self):
        row = array([2, 3, 1, 3, 0, 1, 3, 0, 2, 1, 2])
        col = array([0, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1])
        data = array([6.0, 10.0, 3.0, 9.0, 1.0, 4.0, 11.0, 2.0, 8.0, 5.0, 7.0])
        ij = vstack((row, col))
        csr = csr_matrix((data, ij), (4, 3))
        assert_array_equal(arange(12).reshape(4, 3), csr.toarray())
        csr = csr_matrix(([2 ** 63 + 1, 1], ([0, 1], [0, 1])), dtype=np.uint64)
        dense = array([[2 ** 63 + 1, 0], [0, 1]], dtype=np.uint64)
        assert_array_equal(dense, csr.toarray())

    def test_constructor5(self):
        indptr = array([0, 1, 3, 3])
        indices = array([0, 5, 1, 2])
        data = array([1, 2, 3, 4])
        csr = csr_matrix((data, indices, indptr))
        assert_array_equal(csr.shape, (3, 6))

    def test_constructor6(self):
        indptr = [0, 1, 3, 3]
        indices = [0, 5, 1, 2]
        data = [1, 2, 3, 4]
        csr = csr_matrix((data, indices, indptr))
        assert_array_equal(csr.shape, (3, 6))
        assert_(np.issubdtype(csr.dtype, np.signedinteger))

    def test_constructor_smallcol(self):
        data = arange(6) + 1
        col = array([1, 2, 1, 0, 0, 2], dtype=np.int64)
        ptr = array([0, 2, 4, 6], dtype=np.int64)
        a = csr_matrix((data, col, ptr), shape=(3, 3))
        b = array([[0, 1, 2], [4, 3, 0], [5, 0, 6]], 'd')
        assert_equal(a.indptr.dtype, np.dtype(np.int32))
        assert_equal(a.indices.dtype, np.dtype(np.int32))
        assert_array_equal(a.toarray(), b)

    def test_constructor_largecol(self):
        data = arange(6) + 1
        large = np.iinfo(np.int32).max + 100
        col = array([0, 1, 2, large, large + 1, large + 2], dtype=np.int64)
        ptr = array([0, 2, 4, 6], dtype=np.int64)
        a = csr_matrix((data, col, ptr))
        assert_equal(a.indptr.dtype, np.dtype(np.int64))
        assert_equal(a.indices.dtype, np.dtype(np.int64))
        assert_array_equal(a.shape, (3, max(col) + 1))

    def test_sort_indices(self):
        data = arange(5)
        indices = array([7, 2, 1, 5, 4])
        indptr = array([0, 3, 5])
        asp = csr_matrix((data, indices, indptr), shape=(2, 10))
        bsp = asp.copy()
        asp.sort_indices()
        assert_array_equal(asp.indices, [1, 2, 7, 4, 5])
        assert_array_equal(asp.toarray(), bsp.toarray())

    def test_eliminate_zeros(self):
        data = array([1, 0, 0, 0, 2, 0, 3, 0])
        indices = array([1, 2, 3, 4, 5, 6, 7, 8])
        indptr = array([0, 3, 8])
        asp = csr_matrix((data, indices, indptr), shape=(2, 10))
        bsp = asp.copy()
        asp.eliminate_zeros()
        assert_array_equal(asp.nnz, 3)
        assert_array_equal(asp.data, [1, 2, 3])
        assert_array_equal(asp.toarray(), bsp.toarray())

    def test_ufuncs(self):
        X = csr_matrix(np.arange(20).reshape(4, 5) / 20.0)
        for f in ['sin', 'tan', 'arcsin', 'arctan', 'sinh', 'tanh', 'arcsinh', 'arctanh', 'rint', 'sign', 'expm1', 'log1p', 'deg2rad', 'rad2deg', 'floor', 'ceil', 'trunc', 'sqrt']:
            assert_equal(hasattr(csr_matrix, f), True)
            X2 = getattr(X, f)()
            assert_equal(X.shape, X2.shape)
            assert_array_equal(X.indices, X2.indices)
            assert_array_equal(X.indptr, X2.indptr)
            assert_array_equal(X2.toarray(), getattr(np, f)(X.toarray()))

    def test_unsorted_arithmetic(self):
        data = arange(5)
        indices = array([7, 2, 1, 5, 4])
        indptr = array([0, 3, 5])
        asp = csr_matrix((data, indices, indptr), shape=(2, 10))
        data = arange(6)
        indices = array([8, 1, 5, 7, 2, 4])
        indptr = array([0, 2, 6])
        bsp = csr_matrix((data, indices, indptr), shape=(2, 10))
        assert_equal((asp + bsp).toarray(), asp.toarray() + bsp.toarray())

    def test_fancy_indexing_broadcast(self):
        I = np.array([[1], [2], [3]])
        J = np.array([3, 4, 2])
        np.random.seed(1234)
        D = asmatrix(np.random.rand(5, 7))
        S = self.spcreator(D)
        SIJ = S[I, J]
        if issparse(SIJ):
            SIJ = SIJ.toarray()
        assert_equal(SIJ, D[I, J])

    def test_has_sorted_indices(self):
        """Ensure has_sorted_indices memoizes sorted state for sort_indices"""
        sorted_inds = np.array([0, 1])
        unsorted_inds = np.array([1, 0])
        data = np.array([1, 1])
        indptr = np.array([0, 2])
        M = csr_matrix((data, sorted_inds, indptr)).copy()
        assert_equal(True, M.has_sorted_indices)
        assert isinstance(M.has_sorted_indices, bool)
        M = csr_matrix((data, unsorted_inds, indptr)).copy()
        assert_equal(False, M.has_sorted_indices)
        M.sort_indices()
        assert_equal(True, M.has_sorted_indices)
        assert_array_equal(M.indices, sorted_inds)
        M = csr_matrix((data, unsorted_inds, indptr)).copy()
        M.has_sorted_indices = True
        assert_equal(True, M.has_sorted_indices)
        assert_array_equal(M.indices, unsorted_inds)
        M.sort_indices()
        assert_array_equal(M.indices, unsorted_inds)

    def test_has_canonical_format(self):
        """Ensure has_canonical_format memoizes state for sum_duplicates"""
        M = csr_matrix((np.array([2]), np.array([0]), np.array([0, 1])))
        assert_equal(True, M.has_canonical_format)
        indices = np.array([0, 0])
        data = np.array([1, 1])
        indptr = np.array([0, 2])
        M = csr_matrix((data, indices, indptr)).copy()
        assert_equal(False, M.has_canonical_format)
        assert isinstance(M.has_canonical_format, bool)
        M.sum_duplicates()
        assert_equal(True, M.has_canonical_format)
        assert_equal(1, len(M.indices))
        M = csr_matrix((data, indices, indptr)).copy()
        M.has_canonical_format = True
        assert_equal(True, M.has_canonical_format)
        assert_equal(2, len(M.indices))
        M.sum_duplicates()
        assert_equal(2, len(M.indices))

    def test_scalar_idx_dtype(self):
        indptr = np.zeros(2, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0)
        a = csr_matrix((vals, indices, indptr), shape=(1, 2 ** 31 - 1))
        b = csr_matrix((vals, indices, indptr), shape=(1, 2 ** 31))
        ij = np.zeros((2, 0), dtype=np.int32)
        c = csr_matrix((vals, ij), shape=(1, 2 ** 31 - 1))
        d = csr_matrix((vals, ij), shape=(1, 2 ** 31))
        e = csr_matrix((1, 2 ** 31 - 1))
        f = csr_matrix((1, 2 ** 31))
        assert_equal(a.indptr.dtype, np.int32)
        assert_equal(b.indptr.dtype, np.int64)
        assert_equal(c.indptr.dtype, np.int32)
        assert_equal(d.indptr.dtype, np.int64)
        assert_equal(e.indptr.dtype, np.int32)
        assert_equal(f.indptr.dtype, np.int64)
        for x in [a, b, c, d, e, f]:
            x + x

    def test_binop_explicit_zeros(self):
        a = csr_matrix([0, 1, 0])
        b = csr_matrix([1, 1, 0])
        assert (a + b).nnz == 2
        assert a.multiply(b).nnz == 1