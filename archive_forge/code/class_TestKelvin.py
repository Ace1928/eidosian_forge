import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestKelvin:

    def test_bei(self):
        mbei = special.bei(2)
        assert_almost_equal(mbei, 0.9722916273066613, 5)

    def test_beip(self):
        mbeip = special.beip(2)
        assert_almost_equal(mbeip, 0.9170136133840363, 5)

    def test_ber(self):
        mber = special.ber(2)
        assert_almost_equal(mber, 0.7517341827138082, 5)

    def test_berp(self):
        mberp = special.berp(2)
        assert_almost_equal(mberp, -0.4930671247094391, 5)

    def test_bei_zeros(self):
        bi = special.bei_zeros(5)
        assert_array_almost_equal(bi, array([5.02622, 9.45541, 13.89349, 18.33398, 22.77544]), 4)

    def test_beip_zeros(self):
        bip = special.beip_zeros(5)
        assert_array_almost_equal(bip, array([3.772673304934953, 8.280987849760042, 12.742147523633703, 17.19343175251254, 21.641143941167325]), 8)

    def test_ber_zeros(self):
        ber = special.ber_zeros(5)
        assert_array_almost_equal(ber, array([2.84892, 7.23883, 11.67396, 16.11356, 20.55463]), 4)

    def test_berp_zeros(self):
        brp = special.berp_zeros(5)
        assert_array_almost_equal(brp, array([6.03871, 10.51364, 14.96844, 19.41758, 23.8643]), 4)

    def test_kelvin(self):
        mkelv = special.kelvin(2)
        assert_array_almost_equal(mkelv, (special.ber(2) + special.bei(2) * 1j, special.ker(2) + special.kei(2) * 1j, special.berp(2) + special.beip(2) * 1j, special.kerp(2) + special.keip(2) * 1j), 8)

    def test_kei(self):
        mkei = special.kei(2)
        assert_almost_equal(mkei, -0.20240006776470432, 5)

    def test_keip(self):
        mkeip = special.keip(2)
        assert_almost_equal(mkeip, 0.21980790991960536, 5)

    def test_ker(self):
        mker = special.ker(2)
        assert_almost_equal(mker, -0.04166451399150947, 5)

    def test_kerp(self):
        mkerp = special.kerp(2)
        assert_almost_equal(mkerp, -0.10660096588105264, 5)

    def test_kei_zeros(self):
        kei = special.kei_zeros(5)
        assert_array_almost_equal(kei, array([3.91467, 8.34422, 12.78256, 17.22314, 21.66464]), 4)

    def test_keip_zeros(self):
        keip = special.keip_zeros(5)
        assert_array_almost_equal(keip, array([4.93181, 9.40405, 13.85827, 18.30717, 22.75379]), 4)

    def test_kelvin_zeros(self):
        tmp = special.kelvin_zeros(5)
        berz, beiz, kerz, keiz, berpz, beipz, kerpz, keipz = tmp
        assert_array_almost_equal(berz, array([2.84892, 7.23883, 11.67396, 16.11356, 20.55463]), 4)
        assert_array_almost_equal(beiz, array([5.02622, 9.45541, 13.89349, 18.33398, 22.77544]), 4)
        assert_array_almost_equal(kerz, array([1.71854, 6.12728, 10.56294, 15.00269, 19.44382]), 4)
        assert_array_almost_equal(keiz, array([3.91467, 8.34422, 12.78256, 17.22314, 21.66464]), 4)
        assert_array_almost_equal(berpz, array([6.03871, 10.51364, 14.96844, 19.41758, 23.8643]), 4)
        assert_array_almost_equal(beipz, array([3.77267, 8.28099, 12.74215, 17.19343, 21.64114]), 4)
        assert_array_almost_equal(kerpz, array([2.66584, 7.17212, 11.63218, 16.08312, 20.53068]), 4)
        assert_array_almost_equal(keipz, array([4.93181, 9.40405, 13.85827, 18.30717, 22.75379]), 4)

    def test_ker_zeros(self):
        ker = special.ker_zeros(5)
        assert_array_almost_equal(ker, array([1.71854, 6.12728, 10.56294, 15.00269, 19.44381]), 4)

    def test_kerp_zeros(self):
        kerp = special.kerp_zeros(5)
        assert_array_almost_equal(kerp, array([2.66584, 7.17212, 11.63218, 16.08312, 20.53068]), 4)