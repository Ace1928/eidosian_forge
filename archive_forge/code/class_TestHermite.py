import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestHermite:

    def test_hermite(self):
        H0 = orth.hermite(0)
        H1 = orth.hermite(1)
        H2 = orth.hermite(2)
        H3 = orth.hermite(3)
        H4 = orth.hermite(4)
        H5 = orth.hermite(5)
        assert_array_almost_equal(H0.c, [1], 13)
        assert_array_almost_equal(H1.c, [2, 0], 13)
        assert_array_almost_equal(H2.c, [4, 0, -2], 13)
        assert_array_almost_equal(H3.c, [8, 0, -12, 0], 13)
        assert_array_almost_equal(H4.c, [16, 0, -48, 0, 12], 12)
        assert_array_almost_equal(H5.c, [32, 0, -160, 0, 120, 0], 12)

    def test_hermitenorm(self):
        psub = np.poly1d([1.0 / sqrt(2), 0])
        H0 = orth.hermitenorm(0)
        H1 = orth.hermitenorm(1)
        H2 = orth.hermitenorm(2)
        H3 = orth.hermitenorm(3)
        H4 = orth.hermitenorm(4)
        H5 = orth.hermitenorm(5)
        he0 = orth.hermite(0)(psub)
        he1 = orth.hermite(1)(psub) / sqrt(2)
        he2 = orth.hermite(2)(psub) / 2.0
        he3 = orth.hermite(3)(psub) / (2 * sqrt(2))
        he4 = orth.hermite(4)(psub) / 4.0
        he5 = orth.hermite(5)(psub) / (4.0 * sqrt(2))
        assert_array_almost_equal(H0.c, he0.c, 13)
        assert_array_almost_equal(H1.c, he1.c, 13)
        assert_array_almost_equal(H2.c, he2.c, 13)
        assert_array_almost_equal(H3.c, he3.c, 13)
        assert_array_almost_equal(H4.c, he4.c, 13)
        assert_array_almost_equal(H5.c, he5.c, 13)