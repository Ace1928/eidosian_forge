import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestOrderFilter:

    def test_doc_example(self):
        x = np.arange(25).reshape(5, 5)
        domain = np.identity(3)
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0, 0.0], [0.0, 5.0, 6.0, 7.0, 0.0], [0.0, 10.0, 11.0, 12.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        assert_allclose(order_filter(x, domain, 0), expected)
        expected = np.array([[6.0, 7.0, 8.0, 9.0, 4.0], [11.0, 12.0, 13.0, 14.0, 9.0], [16.0, 17.0, 18.0, 19.0, 14.0], [21.0, 22.0, 23.0, 24.0, 19.0], [20.0, 21.0, 22.0, 23.0, 24.0]])
        assert_allclose(order_filter(x, domain, 2), expected)
        expected = np.array([[0, 1, 2, 3, 0], [5, 6, 7, 8, 3], [10, 11, 12, 13, 8], [15, 16, 17, 18, 13], [0, 15, 16, 17, 18]])
        assert_allclose(order_filter(x, domain, 1), expected)

    def test_medfilt_order_filter(self):
        x = np.arange(25).reshape(5, 5)
        expected = np.array([[0, 1, 2, 3, 0], [1, 6, 7, 8, 4], [6, 11, 12, 13, 9], [11, 16, 17, 18, 14], [0, 16, 17, 18, 0]])
        assert_allclose(medfilt(x, 3), expected)
        assert_allclose(order_filter(x, np.ones((3, 3)), 4), expected)

    def test_order_filter_asymmetric(self):
        x = np.arange(25).reshape(5, 5)
        domain = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        expected = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 5, 6, 7, 8], [0, 10, 11, 12, 13], [0, 15, 16, 17, 18]])
        assert_allclose(order_filter(x, domain, 0), expected)
        expected = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
        assert_allclose(order_filter(x, domain, 1), expected)