import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestSquareForm:
    checked_dtypes = [np.float64, np.float32, np.int32, np.int8, bool]

    def test_squareform_matrix(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_matrix(dtype)

    def test_squareform_vector(self):
        for dtype in self.checked_dtypes:
            self.check_squareform_vector(dtype)

    def check_squareform_matrix(self, dtype):
        A = np.zeros((0, 0), dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)
        A = np.zeros((1, 1), dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (0,))
        assert_equal(rA.dtype, dtype)
        A = np.array([[0, 4.2], [4.2, 0]], dtype=dtype)
        rA = squareform(A)
        assert_equal(rA.shape, (1,))
        assert_equal(rA.dtype, dtype)
        assert_array_equal(rA, np.array([4.2], dtype=dtype))

    def check_squareform_vector(self, dtype):
        v = np.zeros((0,), dtype=dtype)
        rv = squareform(v)
        assert_equal(rv.shape, (1, 1))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, [[0]])
        v = np.array([8.3], dtype=dtype)
        rv = squareform(v)
        assert_equal(rv.shape, (2, 2))
        assert_equal(rv.dtype, dtype)
        assert_array_equal(rv, np.array([[0, 8.3], [8.3, 0]], dtype=dtype))

    def test_squareform_multi_matrix(self):
        for n in range(2, 5):
            self.check_squareform_multi_matrix(n)

    def check_squareform_multi_matrix(self, n):
        X = np.random.rand(n, 4)
        Y = wpdist_no_const(X)
        assert_equal(len(Y.shape), 1)
        A = squareform(Y)
        Yr = squareform(A)
        s = A.shape
        k = 0
        if verbose >= 3:
            print(A.shape, Y.shape, Yr.shape)
        assert_equal(len(s), 2)
        assert_equal(len(Yr.shape), 1)
        assert_equal(s[0], s[1])
        for i in range(0, s[0]):
            for j in range(i + 1, s[1]):
                if i != j:
                    assert_equal(A[i, j], Y[k])
                    k += 1
                else:
                    assert_equal(A[i, j], 0)