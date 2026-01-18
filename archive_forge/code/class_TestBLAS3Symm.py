import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestBLAS3Symm:

    def setup_method(self):
        self.a = np.array([[1.0, 2.0], [0.0, 1.0]])
        self.b = np.array([[1.0, 0.0, 3.0], [0.0, -1.0, 2.0]])
        self.c = np.ones((2, 3))
        self.t = np.array([[2.0, -1.0, 8.0], [3.0, 0.0, 9.0]])

    def test_symm(self):
        for f in _get_func('symm'):
            res = f(a=self.a, b=self.b, c=self.c, alpha=1.0, beta=1.0)
            assert_array_almost_equal(res, self.t)
            res = f(a=self.a.T, b=self.b, lower=1, c=self.c, alpha=1.0, beta=1.0)
            assert_array_almost_equal(res, self.t)
            res = f(a=self.a, b=self.b.T, side=1, c=self.c.T, alpha=1.0, beta=1.0)
            assert_array_almost_equal(res, self.t.T)

    def test_summ_wrong_side(self):
        f = getattr(fblas, 'dsymm', None)
        if f is not None:
            assert_raises(Exception, f, **{'a': self.a, 'b': self.b, 'alpha': 1, 'side': 1})

    def test_symm_wrong_uplo(self):
        """SYMM only considers the upper/lower part of A. Hence setting
        wrong value for `lower` (default is lower=0, meaning upper triangle)
        gives a wrong result.
        """
        f = getattr(fblas, 'dsymm', None)
        if f is not None:
            res = f(a=self.a, b=self.b, c=self.c, alpha=1.0, beta=1.0)
            assert np.allclose(res, self.t)
            res = f(a=self.a, b=self.b, lower=1, c=self.c, alpha=1.0, beta=1.0)
            assert not np.allclose(res, self.t)