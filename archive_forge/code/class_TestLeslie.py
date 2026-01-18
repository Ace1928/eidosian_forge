import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
class TestLeslie:

    def test_bad_shapes(self):
        assert_raises(ValueError, leslie, [[1, 1], [2, 2]], [3, 4, 5])
        assert_raises(ValueError, leslie, [3, 4, 5], [[1, 1], [2, 2]])
        assert_raises(ValueError, leslie, [1, 2], [1, 2])
        assert_raises(ValueError, leslie, [1], [])

    def test_basic(self):
        a = leslie([1, 2, 3], [0.25, 0.5])
        expected = array([[1.0, 2.0, 3.0], [0.25, 0.0, 0.0], [0.0, 0.5, 0.0]])
        assert_array_equal(a, expected)