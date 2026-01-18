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
class TestEllipCarlson:
    """Test for Carlson elliptic integrals ellipr[cdfgj].
    The special values used in these tests can be found in Sec. 3 of Carlson
    (1994), https://arxiv.org/abs/math/9409227
    """

    def test_elliprc(self):
        assert_allclose(elliprc(1, 1), 1)
        assert elliprc(1, inf) == 0.0
        assert isnan(elliprc(1, 0))
        assert elliprc(1, complex(1, inf)) == 0.0
        args = array([[0.0, 0.25], [2.25, 2.0], [0.0, 1j], [-1j, 1j], [0.25, -2.0], [1j, -1.0]])
        expected_results = array([np.pi, np.log(2.0), 1.1107207345396 * (1.0 - 1j), 1.2260849569072 - 0.34471136988768j, np.log(2.0) / 3.0, 0.77778596920447 + 0.19832484993429j])
        for i, arr in enumerate(args):
            assert_allclose(elliprc(*arr), expected_results[i])

    def test_elliprd(self):
        assert_allclose(elliprd(1, 1, 1), 1)
        assert_allclose(elliprd(0, 2, 1) / 3.0, 0.5990701173677961)
        assert elliprd(1, 1, inf) == 0.0
        assert np.isinf(elliprd(1, 1, 0))
        assert np.isinf(elliprd(1, 1, complex(0, 0)))
        assert np.isinf(elliprd(0, 1, complex(0, 0)))
        assert isnan(elliprd(1, 1, -np.finfo(np.float64).tiny / 2.0))
        assert isnan(elliprd(1, 1, complex(-1, 0)))
        args = array([[0.0, 2.0, 1.0], [2.0, 3.0, 4.0], [1j, -1j, 2.0], [0.0, 1j, -1j], [0.0, -1.0 + 1j, 1j], [-2.0 - 1j, -1j, -1.0 + 1j]])
        expected_results = array([1.7972103521034, 0.16510527294261, 0.6593385415422, 1.270819627191 + 2.7811120159521j, -1.8577235439239 - 0.96193450888839j, 1.8249027393704 - 1.2218475784827j])
        for i, arr in enumerate(args):
            assert_allclose(elliprd(*arr), expected_results[i])

    def test_elliprf(self):
        assert_allclose(elliprf(1, 1, 1), 1)
        assert_allclose(elliprf(0, 1, 2), 1.3110287771460598)
        assert elliprf(1, inf, 1) == 0.0
        assert np.isinf(elliprf(0, 1, 0))
        assert isnan(elliprf(1, 1, -1))
        assert elliprf(complex(inf), 0, 1) == 0.0
        assert isnan(elliprf(1, 1, complex(-inf, 1)))
        args = array([[1.0, 2.0, 0.0], [1j, -1j, 0.0], [0.5, 1.0, 0.0], [-1.0 + 1j, 1j, 0.0], [2.0, 3.0, 4.0], [1j, -1j, 2.0], [-1.0 + 1j, 1j, 1.0 - 1j]])
        expected_results = array([1.3110287771461, 1.8540746773014, 1.8540746773014, 0.79612586584234 - 1.2138566698365j, 0.58408284167715, 1.0441445654064, 0.93912050218619 - 0.53296252018635j])
        for i, arr in enumerate(args):
            assert_allclose(elliprf(*arr), expected_results[i])

    def test_elliprg(self):
        assert_allclose(elliprg(1, 1, 1), 1)
        assert_allclose(elliprg(0, 0, 1), 0.5)
        assert_allclose(elliprg(0, 0, 0), 0)
        assert np.isinf(elliprg(1, inf, 1))
        assert np.isinf(elliprg(complex(inf), 1, 1))
        args = array([[0.0, 16.0, 16.0], [2.0, 3.0, 4.0], [0.0, 1j, -1j], [-1.0 + 1j, 1j, 0.0], [-1j, -1.0 + 1j, 1j], [0.0, 0.0796, 4.0]])
        expected_results = array([np.pi, 1.7255030280692, 0.42360654239699, 0.44660591677018 + 0.70768352357515j, 0.36023392184473 + 0.40348623401722j, 1.0284758090288])
        for i, arr in enumerate(args):
            assert_allclose(elliprg(*arr), expected_results[i])

    def test_elliprj(self):
        assert_allclose(elliprj(1, 1, 1, 1), 1)
        assert elliprj(1, 1, inf, 1) == 0.0
        assert isnan(elliprj(1, 0, 0, 0))
        assert isnan(elliprj(-1, 1, 1, 1))
        assert elliprj(1, 1, 1, inf) == 0.0
        args = array([[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, -1.0 + 1j], [1j, -1j, 0.0, 2.0], [-1.0 + 1j, -1.0 - 1j, 1.0, 2.0], [1j, -1j, 0.0, 1.0 - 1j], [-1.0 + 1j, -1.0 - 1j, 1.0, -3.0 + 1j], [2.0, 3.0, 4.0, -0.5], [2.0, 3.0, 4.0, -5.0]])
        expected_results = array([0.77688623778582, 0.14297579667157, 0.13613945827771 - 0.38207561624427j, 1.6490011662711, 0.9414835884122, 1.8260115229009 + 1.2290661908643j, -0.61127970812028 - 1.0684038390007j, 0.24723819703052, -0.12711230042964])
        for i, arr in enumerate(args):
            assert_allclose(elliprj(*arr), expected_results[i])

    @pytest.mark.xfail(reason='Insufficient accuracy on 32-bit')
    def test_elliprj_hard(self):
        assert_allclose(elliprj(6.483625725195452e-08, 1.1649136528196886e-27, 36767340167168.0, 0.493704617023468), 8.634269206442419e-06, rtol=5e-15, atol=1e-20)
        assert_allclose(elliprj(14.375105857849121, 9.993988969725365e-11, 1.72844262269944e-26, 5.898871222598245e-06), 829774.1424801627, rtol=5e-15, atol=1e-20)