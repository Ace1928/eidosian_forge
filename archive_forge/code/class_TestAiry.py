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
class TestAiry:

    def test_airy(self):
        x = special.airy(0.99)
        assert_array_almost_equal(x, array([0.13689066, -0.16050153, 1.19815925, 0.92046818]), 8)
        x = special.airy(0.41)
        assert_array_almost_equal(x, array([0.25238916, -0.23480512, 0.80686202, 0.51053919]), 8)
        x = special.airy(-0.36)
        assert_array_almost_equal(x, array([0.44508477, -0.23186773, 0.44939534, 0.48105354]), 8)

    def test_airye(self):
        a = special.airye(0.01)
        b = special.airy(0.01)
        b1 = [None] * 4
        for n in range(2):
            b1[n] = b[n] * exp(2.0 / 3.0 * 0.01 * sqrt(0.01))
        for n in range(2, 4):
            b1[n] = b[n] * exp(-abs(real(2.0 / 3.0 * 0.01 * sqrt(0.01))))
        assert_array_almost_equal(a, b1, 6)

    def test_bi_zeros(self):
        bi = special.bi_zeros(2)
        bia = (array([-1.17371322, -3.271093]), array([-2.29443968, -4.07315509]), array([-0.45494438, 0.39652284]), array([0.60195789, -0.76031014]))
        assert_array_almost_equal(bi, bia, 4)
        bi = special.bi_zeros(5)
        assert_array_almost_equal(bi[0], array([-1.173713222709127, -3.271093302836352, -4.830737841662016, -6.169852128310251, -7.376762079367764]), 11)
        assert_array_almost_equal(bi[1], array([-2.294439682614122, -4.073155089071828, -5.512395729663599, -6.781294445990305, -7.940178689168587]), 10)
        assert_array_almost_equal(bi[2], array([-0.454944383639657, 0.396522836094465, -0.367969161486959, 0.349499116831805, -0.336026240133662]), 11)
        assert_array_almost_equal(bi[3], array([0.601957887976239, -0.760310141492801, 0.836991012619261, -0.88947990142654, 0.929983638568022]), 10)

    def test_ai_zeros(self):
        ai = special.ai_zeros(1)
        assert_array_almost_equal(ai, (array([-2.33810741]), array([-1.01879297]), array([0.5357]), array([0.7012])), 4)

    def test_ai_zeros_big(self):
        z, zp, ai_zpx, aip_zx = special.ai_zeros(50000)
        ai_z, aip_z, _, _ = special.airy(z)
        ai_zp, aip_zp, _, _ = special.airy(zp)
        ai_envelope = 1 / abs(z) ** (1.0 / 4)
        aip_envelope = abs(zp) ** (1.0 / 4)
        assert_allclose(ai_zpx, ai_zp, rtol=1e-10)
        assert_allclose(aip_zx, aip_z, rtol=1e-10)
        assert_allclose(ai_z / ai_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(aip_zp / aip_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(z[:6], [-2.3381074105, -4.0879494441, -5.5205598281, -6.7867080901, -7.9441335871, -9.0226508533], rtol=1e-10)
        assert_allclose(zp[:6], [-1.0187929716, -3.2481975822, -4.8200992112, -6.1633073556, -7.372177255, -8.488486734], rtol=1e-10)

    def test_bi_zeros_big(self):
        z, zp, bi_zpx, bip_zx = special.bi_zeros(50000)
        _, _, bi_z, bip_z = special.airy(z)
        _, _, bi_zp, bip_zp = special.airy(zp)
        bi_envelope = 1 / abs(z) ** (1.0 / 4)
        bip_envelope = abs(zp) ** (1.0 / 4)
        assert_allclose(bi_zpx, bi_zp, rtol=1e-10)
        assert_allclose(bip_zx, bip_z, rtol=1e-10)
        assert_allclose(bi_z / bi_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(bip_zp / bip_envelope, 0, atol=1e-10, rtol=0)
        assert_allclose(z[:6], [-1.1737132227, -3.2710933028, -4.8307378417, -6.1698521283, -7.3767620794, -8.4919488465], rtol=1e-10)
        assert_allclose(zp[:6], [-2.2944396826, -4.0731550891, -5.5123957297, -6.781294446, -7.9401786892, -9.0195833588], rtol=1e-10)