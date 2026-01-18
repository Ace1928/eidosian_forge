import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestToeplitz:

    def test_basic(self):
        y = toeplitz([1, 2, 3])
        assert_array_equal(y, [[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        y = toeplitz([1, 2, 3], [1, 4, 5])
        assert_array_equal(y, [[1, 4, 5], [2, 1, 4], [3, 2, 1]])

    def test_complex_01(self):
        data = (1.0 + arange(3.0)) * (1.0 + 1j)
        x = copy(data)
        t = toeplitz(x)
        assert_array_equal(x, data)
        col0 = t[:, 0]
        assert_array_equal(col0, data)
        assert_array_equal(t[0, 1:], data[1:].conj())

    def test_scalar_00(self):
        """Scalar arguments still produce a 2D array."""
        t = toeplitz(10)
        assert_array_equal(t, [[10]])
        t = toeplitz(10, 20)
        assert_array_equal(t, [[10]])

    def test_scalar_01(self):
        c = array([1, 2, 3])
        t = toeplitz(c, 1)
        assert_array_equal(t, [[1], [2], [3]])

    def test_scalar_02(self):
        c = array([1, 2, 3])
        t = toeplitz(c, array(1))
        assert_array_equal(t, [[1], [2], [3]])

    def test_scalar_03(self):
        c = array([1, 2, 3])
        t = toeplitz(c, array([1]))
        assert_array_equal(t, [[1], [2], [3]])

    def test_scalar_04(self):
        r = array([10, 2, 3])
        t = toeplitz(1, r)
        assert_array_equal(t, [[1, 2, 3]])