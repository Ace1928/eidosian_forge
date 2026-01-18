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
class TestDOK(sparse_test_class(minmax=False, nnz_axis=False)):
    spcreator = dok_matrix
    math_dtypes = [np.int_, np.float64, np.complex128]

    def test_mult(self):
        A = dok_matrix((10, 10))
        A[0, 3] = 10
        A[5, 6] = 20
        D = A * A.T
        E = A * A.H
        assert_array_equal(D.toarray(), E.toarray())

    def test_add_nonzero(self):
        A = self.spcreator((3, 2))
        A[0, 1] = -10
        A[2, 0] = 20
        A = A + 10
        B = array([[10, 0], [10, 10], [30, 10]])
        assert_array_equal(A.toarray(), B)
        A = A + 1j
        B = B + 1j
        assert_array_equal(A.toarray(), B)

    def test_dok_divide_scalar(self):
        A = self.spcreator((3, 2))
        A[0, 1] = -10
        A[2, 0] = 20
        assert_array_equal((A / 1j).toarray(), A.toarray() / 1j)
        assert_array_equal((A / 9).toarray(), A.toarray() / 9)

    def test_convert(self):
        m, n = (6, 7)
        a = dok_matrix((m, n))
        a[2, 1] = 1
        a[0, 2] = 2
        a[3, 1] = 3
        a[1, 5] = 4
        a[4, 3] = 5
        a[4, 2] = 6
        assert_array_equal(a.toarray()[:, n - 1], zeros(m))
        csc = a.tocsc()
        assert_array_equal(csc.toarray()[:, n - 1], zeros(m))
        m, n = (n, m)
        b = a.transpose()
        assert_equal(b.shape, (m, n))
        assert_array_equal(b.toarray()[m - 1, :], zeros(n))
        csr = b.tocsr()
        assert_array_equal(csr.toarray()[m - 1, :], zeros(n))

    def test_ctor(self):
        assert_raises(TypeError, dok_matrix)
        b = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 3]], 'd')
        A = dok_matrix(b)
        assert_equal(b.dtype, A.dtype)
        assert_equal(A.toarray(), b)
        c = csr_matrix(b)
        assert_equal(A.toarray(), c.toarray())
        data = [[0, 1, 2], [3, 0, 0]]
        d = dok_matrix(data, dtype=np.float32)
        assert_equal(d.dtype, np.float32)
        da = d.toarray()
        assert_equal(da.dtype, np.float32)
        assert_array_equal(da, data)

    def test_ticket1160(self):
        a = dok_matrix((3, 3))
        a[0, 0] = 0
        assert_((0, 0) not in a.keys(), 'Unexpected entry (0,0) in keys')
        b = dok_matrix((3, 3))
        b[:, 0] = 0
        assert_(len(b.keys()) == 0, 'Unexpected entries in keys')