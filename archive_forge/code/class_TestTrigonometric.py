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
class TestTrigonometric:

    def test_cbrt(self):
        cb = special.cbrt(27)
        cbrl = 27 ** (1.0 / 3.0)
        assert_approx_equal(cb, cbrl)

    def test_cbrtmore(self):
        cb1 = special.cbrt(27.9)
        cbrl1 = 27.9 ** (1.0 / 3.0)
        assert_almost_equal(cb1, cbrl1, 8)

    def test_cosdg(self):
        cdg = special.cosdg(90)
        cdgrl = cos(pi / 2.0)
        assert_almost_equal(cdg, cdgrl, 8)

    def test_cosdgmore(self):
        cdgm = special.cosdg(30)
        cdgmrl = cos(pi / 6.0)
        assert_almost_equal(cdgm, cdgmrl, 8)

    def test_cosm1(self):
        cs = (special.cosm1(0), special.cosm1(0.3), special.cosm1(pi / 10))
        csrl = (cos(0) - 1, cos(0.3) - 1, cos(pi / 10) - 1)
        assert_array_almost_equal(cs, csrl, 8)

    def test_cotdg(self):
        ct = special.cotdg(30)
        ctrl = tan(pi / 6.0) ** (-1)
        assert_almost_equal(ct, ctrl, 8)

    def test_cotdgmore(self):
        ct1 = special.cotdg(45)
        ctrl1 = tan(pi / 4.0) ** (-1)
        assert_almost_equal(ct1, ctrl1, 8)

    def test_specialpoints(self):
        assert_almost_equal(special.cotdg(45), 1.0, 14)
        assert_almost_equal(special.cotdg(-45), -1.0, 14)
        assert_almost_equal(special.cotdg(90), 0.0, 14)
        assert_almost_equal(special.cotdg(-90), 0.0, 14)
        assert_almost_equal(special.cotdg(135), -1.0, 14)
        assert_almost_equal(special.cotdg(-135), 1.0, 14)
        assert_almost_equal(special.cotdg(225), 1.0, 14)
        assert_almost_equal(special.cotdg(-225), -1.0, 14)
        assert_almost_equal(special.cotdg(270), 0.0, 14)
        assert_almost_equal(special.cotdg(-270), 0.0, 14)
        assert_almost_equal(special.cotdg(315), -1.0, 14)
        assert_almost_equal(special.cotdg(-315), 1.0, 14)
        assert_almost_equal(special.cotdg(765), 1.0, 14)

    def test_sinc(self):
        assert_array_equal(special.sinc([0]), 1)
        assert_equal(special.sinc(0.0), 1.0)

    def test_sindg(self):
        sn = special.sindg(90)
        assert_equal(sn, 1.0)

    def test_sindgmore(self):
        snm = special.sindg(30)
        snmrl = sin(pi / 6.0)
        assert_almost_equal(snm, snmrl, 8)
        snm1 = special.sindg(45)
        snmrl1 = sin(pi / 4.0)
        assert_almost_equal(snm1, snmrl1, 8)