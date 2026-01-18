import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose,
import pytest
from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad
class TestSphericalOld:

    def test_sph_in(self):
        i1n = np.empty((2, 2))
        x = 0.2
        i1n[0][0] = spherical_in(0, x)
        i1n[0][1] = spherical_in(1, x)
        i1n[1][0] = spherical_in(0, x, derivative=True)
        i1n[1][1] = spherical_in(1, x, derivative=True)
        inp0 = i1n[0][1]
        inp1 = i1n[0][0] - 2.0 / 0.2 * i1n[0][1]
        assert_array_almost_equal(i1n[0], np.array([1.00668001270547, 0.06693371456802955]), 12)
        assert_array_almost_equal(i1n[1], [inp0, inp1], 12)

    def test_sph_in_kn_order0(self):
        x = 1.0
        sph_i0 = np.empty((2,))
        sph_i0[0] = spherical_in(0, x)
        sph_i0[1] = spherical_in(0, x, derivative=True)
        sph_i0_expected = np.array([np.sinh(x) / x, np.cosh(x) / x - np.sinh(x) / x ** 2])
        assert_array_almost_equal(r_[sph_i0], sph_i0_expected)
        sph_k0 = np.empty((2,))
        sph_k0[0] = spherical_kn(0, x)
        sph_k0[1] = spherical_kn(0, x, derivative=True)
        sph_k0_expected = np.array([0.5 * pi * exp(-x) / x, -0.5 * pi * exp(-x) * (1 / x + 1 / x ** 2)])
        assert_array_almost_equal(r_[sph_k0], sph_k0_expected)

    def test_sph_jn(self):
        s1 = np.empty((2, 3))
        x = 0.2
        s1[0][0] = spherical_jn(0, x)
        s1[0][1] = spherical_jn(1, x)
        s1[0][2] = spherical_jn(2, x)
        s1[1][0] = spherical_jn(0, x, derivative=True)
        s1[1][1] = spherical_jn(1, x, derivative=True)
        s1[1][2] = spherical_jn(2, x, derivative=True)
        s10 = -s1[0][1]
        s11 = s1[0][0] - 2.0 / 0.2 * s1[0][1]
        s12 = s1[0][1] - 3.0 / 0.2 * s1[0][2]
        assert_array_almost_equal(s1[0], [0.9933466539753061, 0.06640038067032224, 0.0026590560795273855], 12)
        assert_array_almost_equal(s1[1], [s10, s11, s12], 12)

    def test_sph_kn(self):
        kn = np.empty((2, 3))
        x = 0.2
        kn[0][0] = spherical_kn(0, x)
        kn[0][1] = spherical_kn(1, x)
        kn[0][2] = spherical_kn(2, x)
        kn[1][0] = spherical_kn(0, x, derivative=True)
        kn[1][1] = spherical_kn(1, x, derivative=True)
        kn[1][2] = spherical_kn(2, x, derivative=True)
        kn0 = -kn[0][1]
        kn1 = -kn[0][0] - 2.0 / 0.2 * kn[0][1]
        kn2 = -kn[0][1] - 3.0 / 0.2 * kn[0][2]
        assert_array_almost_equal(kn[0], [6.430296297844567, 38.5817777870674, 585.1569631038556], 12)
        assert_array_almost_equal(kn[1], [kn0, kn1, kn2], 9)

    def test_sph_yn(self):
        sy1 = spherical_yn(2, 0.2)
        sy2 = spherical_yn(0, 0.2)
        assert_almost_equal(sy1, -377.52483, 5)
        assert_almost_equal(sy2, -4.9003329, 5)
        sphpy = (spherical_yn(0, 0.2) - 2 * spherical_yn(2, 0.2)) / 3
        sy3 = spherical_yn(1, 0.2, derivative=True)
        assert_almost_equal(sy3, sphpy, 4)