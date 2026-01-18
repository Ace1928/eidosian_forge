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
class TestLIL(sparse_test_class(minmax=False)):
    spcreator = lil_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]

    def test_dot(self):
        A = zeros((10, 10), np.complex128)
        A[0, 3] = 10
        A[5, 6] = 20j
        B = lil_matrix((10, 10), dtype=np.complex128)
        B[0, 3] = 10
        B[5, 6] = 20j
        if platform.machine() != 'ppc64le':
            assert_array_equal(A @ A.T, (B * B.T).toarray())
        assert_array_equal(A @ A.conjugate().T, (B * B.conjugate().T).toarray())

    def test_scalar_mul(self):
        x = lil_matrix((3, 3))
        x[0, 0] = 2
        x = x * 2
        assert_equal(x[0, 0], 4)
        x = x * 0
        assert_equal(x[0, 0], 0)

    def test_truediv_scalar(self):
        A = self.spcreator((3, 2))
        A[0, 1] = -10
        A[2, 0] = 20
        assert_array_equal((A / 1j).toarray(), A.toarray() / 1j)
        assert_array_equal((A / 9).toarray(), A.toarray() / 9)

    def test_inplace_ops(self):
        A = lil_matrix([[0, 2, 3], [4, 0, 6]])
        B = lil_matrix([[0, 1, 0], [0, 2, 3]])
        data = {'add': (B, A + B), 'sub': (B, A - B), 'mul': (3, A * 3)}
        for op, (other, expected) in data.items():
            result = A.copy()
            getattr(result, '__i%s__' % op)(other)
            assert_array_equal(result.toarray(), expected.toarray())
        A = lil_matrix((1, 3), dtype=np.dtype('float64'))
        B = array([0.1, 0.1, 0.1])
        A[0, :] += B
        assert_array_equal(A[0, :].toarray().squeeze(), B)

    def test_lil_iteration(self):
        row_data = [[1, 2, 3], [4, 5, 6]]
        B = lil_matrix(array(row_data))
        for r, row in enumerate(B):
            assert_array_equal(row.toarray(), array(row_data[r], ndmin=2))

    def test_lil_from_csr(self):
        B = lil_matrix((10, 10))
        B[0, 3] = 10
        B[5, 6] = 20
        B[8, 3] = 30
        B[3, 8] = 40
        B[8, 9] = 50
        C = B.tocsr()
        D = lil_matrix(C)
        assert_array_equal(C.toarray(), D.toarray())

    def test_fancy_indexing_lil(self):
        M = asmatrix(arange(25).reshape(5, 5))
        A = lil_matrix(M)
        assert_equal(A[array([1, 2, 3]), 2:3].toarray(), M[array([1, 2, 3]), 2:3])

    def test_point_wise_multiply(self):
        l = lil_matrix((4, 3))
        l[0, 0] = 1
        l[1, 1] = 2
        l[2, 2] = 3
        l[3, 1] = 4
        m = lil_matrix((4, 3))
        m[0, 0] = 1
        m[0, 1] = 2
        m[2, 2] = 3
        m[3, 1] = 4
        m[3, 2] = 4
        assert_array_equal(l.multiply(m).toarray(), m.multiply(l).toarray())
        assert_array_equal(l.multiply(m).toarray(), [[1, 0, 0], [0, 0, 0], [0, 0, 9], [0, 16, 0]])

    def test_lil_multiply_removal(self):
        a = lil_matrix(np.ones((3, 3)))
        a *= 2.0
        a[0, :] = 0