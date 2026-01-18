import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestCplxPair:

    def test_trivial_input(self):
        assert_equal(_cplxpair([]).size, 0)
        assert_equal(_cplxpair(1), 1)

    def test_output_order(self):
        assert_allclose(_cplxpair([1 + 1j, 1 - 1j]), [1 - 1j, 1 + 1j])
        a = [1 + 1j, 1 + 1j, 1, 1 - 1j, 1 - 1j, 2]
        b = [1 - 1j, 1 + 1j, 1 - 1j, 1 + 1j, 1, 2]
        assert_allclose(_cplxpair(a), b)
        z = np.exp(2j * pi * array([4, 3, 5, 2, 6, 1, 0]) / 7)
        z1 = np.copy(z)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        x = np.random.rand(10000) + 1j * np.random.rand(10000)
        y = x.conj()
        z = np.random.rand(10000)
        x = np.concatenate((x, y, z))
        np.random.shuffle(x)
        c = _cplxpair(x)
        assert_allclose(c[0:20000:2], np.conj(c[1:20000:2]))
        assert_allclose(c[0:20000:2].real, np.sort(c[0:20000:2].real))
        assert_allclose(c[20000:], np.sort(c[20000:]))

    def test_real_integer_input(self):
        assert_array_equal(_cplxpair([2, 0, 1]), [0, 1, 2])

    def test_tolerances(self):
        eps = spacing(1)
        assert_allclose(_cplxpair([1j, -1j, 1 + 1j * eps], tol=2 * eps), [-1j, 1j, 1 + 1j * eps])
        assert_allclose(_cplxpair([-eps + 1j, +eps - 1j]), [-1j, +1j])
        assert_allclose(_cplxpair([+eps + 1j, -eps - 1j]), [-1j, +1j])
        assert_allclose(_cplxpair([+1j, -1j]), [-1j, +1j])

    def test_unmatched_conjugates(self):
        assert_raises(ValueError, _cplxpair, [1 + 3j, 1 - 3j, 1 + 2j])
        assert_raises(ValueError, _cplxpair, [1 + 3j, 1 - 3j, 1 + 2j, 1 - 3j])
        assert_raises(ValueError, _cplxpair, [1 + 3j, 1 - 3j, 1 + 3j])
        assert_raises(ValueError, _cplxpair, [4 + 5j, 4 + 5j])
        assert_raises(ValueError, _cplxpair, [1 - 7j, 1 - 7j])
        assert_raises(ValueError, _cplxpair, [1 + 3j])
        assert_raises(ValueError, _cplxpair, [1 - 3j])