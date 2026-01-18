import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
@dep_filter
class TestTriu:

    def test_basic(self):
        a = (100 * get_mat(5)).astype('l')
        b = a.copy()
        for k in range(5):
            for l in range(k + 1, 5):
                b[l, k] = 0
        assert_equal(triu(a), b)

    def test_diag(self):
        a = (100 * get_mat(5)).astype('f')
        b = a.copy()
        for k in range(5):
            for l in range(max((k - 1, 0)), 5):
                b[l, k] = 0
        assert_equal(triu(a, k=2), b)
        b = a.copy()
        for k in range(5):
            for l in range(k + 3, 5):
                b[l, k] = 0
        assert_equal(triu(a, k=-2), b)