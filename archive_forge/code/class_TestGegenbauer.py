import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestGegenbauer:

    def test_gegenbauer(self):
        a = 5 * np.random.random() - 0.5
        if np.any(a == 0):
            a = -0.2
        Ca0 = orth.gegenbauer(0, a)
        Ca1 = orth.gegenbauer(1, a)
        Ca2 = orth.gegenbauer(2, a)
        Ca3 = orth.gegenbauer(3, a)
        Ca4 = orth.gegenbauer(4, a)
        Ca5 = orth.gegenbauer(5, a)
        assert_array_almost_equal(Ca0.c, array([1]), 13)
        assert_array_almost_equal(Ca1.c, array([2 * a, 0]), 13)
        assert_array_almost_equal(Ca2.c, array([2 * a * (a + 1), 0, -a]), 13)
        assert_array_almost_equal(Ca3.c, array([4 * sc.poch(a, 3), 0, -6 * a * (a + 1), 0]) / 3.0, 11)
        assert_array_almost_equal(Ca4.c, array([4 * sc.poch(a, 4), 0, -12 * sc.poch(a, 3), 0, 3 * a * (a + 1)]) / 6.0, 11)
        assert_array_almost_equal(Ca5.c, array([4 * sc.poch(a, 5), 0, -20 * sc.poch(a, 4), 0, 15 * sc.poch(a, 3), 0]) / 15.0, 11)