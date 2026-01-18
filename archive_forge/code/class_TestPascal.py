import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestPascal:
    cases = [(1, array([[1]]), array([[1]])), (2, array([[1, 1], [1, 2]]), array([[1, 0], [1, 1]])), (3, array([[1, 1, 1], [1, 2, 3], [1, 3, 6]]), array([[1, 0, 0], [1, 1, 0], [1, 2, 1]])), (4, array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 6, 10], [1, 4, 10, 20]]), array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 1, 0], [1, 3, 3, 1]]))]

    def check_case(self, n, sym, low):
        assert_array_equal(pascal(n), sym)
        assert_array_equal(pascal(n, kind='lower'), low)
        assert_array_equal(pascal(n, kind='upper'), low.T)
        assert_array_almost_equal(pascal(n, exact=False), sym)
        assert_array_almost_equal(pascal(n, exact=False, kind='lower'), low)
        assert_array_almost_equal(pascal(n, exact=False, kind='upper'), low.T)

    def test_cases(self):
        for n, sym, low in self.cases:
            self.check_case(n, sym, low)

    def test_big(self):
        p = pascal(50)
        assert p[-1, -1] == comb(98, 49, exact=True)

    def test_threshold(self):
        p = pascal(34)
        assert_equal(2 * p.item(-1, -2), p.item(-1, -1), err_msg='n = 34')
        p = pascal(35)
        assert_equal(2.0 * p.item(-1, -2), 1.0 * p.item(-1, -1), err_msg='n = 35')