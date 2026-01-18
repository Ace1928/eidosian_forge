import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestMatrixNorms:

    def test_matrix_norms(self):
        np.random.seed(1234)
        for n, m in ((1, 1), (1, 3), (3, 1), (4, 4), (4, 5), (5, 4)):
            for t in (np.float32, np.float64, np.complex64, np.complex128, np.int64):
                A = 10 * np.random.randn(n, m).astype(t)
                if np.issubdtype(A.dtype, np.complexfloating):
                    A = (A + 10j * np.random.randn(n, m)).astype(t)
                    t_high = np.complex128
                else:
                    t_high = np.float64
                for order in (None, 'fro', 1, -1, 2, -2, np.inf, -np.inf):
                    actual = norm(A, ord=order)
                    desired = np.linalg.norm(A, ord=order)
                    if not np.allclose(actual, desired):
                        desired = np.linalg.norm(A.astype(t_high), ord=order)
                        assert_allclose(actual, desired)

    def test_axis_kwd(self):
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        b = norm(a, ord=np.inf, axis=(1, 0))
        c = norm(np.swapaxes(a, 0, 1), ord=np.inf, axis=(0, 1))
        d = norm(a, ord=1, axis=(0, 1))
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        assert_(b.shape == c.shape == d.shape)
        b = norm(a, ord=1, axis=(1, 0))
        c = norm(np.swapaxes(a, 0, 1), ord=1, axis=(0, 1))
        d = norm(a, ord=np.inf, axis=(0, 1))
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        assert_(b.shape == c.shape == d.shape)

    def test_keepdims_kwd(self):
        a = np.arange(120, dtype='d').reshape(2, 3, 4, 5)
        b = norm(a, ord=np.inf, axis=(1, 0), keepdims=True)
        c = norm(a, ord=1, axis=(0, 1), keepdims=True)
        assert_allclose(b, c)
        assert_(b.shape == c.shape)