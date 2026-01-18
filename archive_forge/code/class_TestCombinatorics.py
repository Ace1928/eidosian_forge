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
class TestCombinatorics:

    def test_comb(self):
        assert_array_almost_equal(special.comb([10, 10], [3, 4]), [120.0, 210.0])
        assert_almost_equal(special.comb(10, 3), 120.0)
        assert_equal(special.comb(10, 3, exact=True), 120)
        assert_equal(special.comb(10, 3, exact=True, repetition=True), 220)
        assert_allclose([special.comb(20, k, exact=True) for k in range(21)], special.comb(20, list(range(21))), atol=1e-15)
        ii = np.iinfo(int).max + 1
        assert_equal(special.comb(ii, ii - 1, exact=True), ii)
        expected = 100891344545564193334812497256
        assert special.comb(100, 50, exact=True) == expected

    @pytest.mark.parametrize('repetition', [True, False])
    @pytest.mark.parametrize('legacy', [True, False, _NoValue])
    @pytest.mark.parametrize('k', [3.5, 3])
    @pytest.mark.parametrize('N', [4.5, 4])
    def test_comb_legacy(self, N, k, legacy, repetition):
        if legacy is not _NoValue:
            with pytest.warns(DeprecationWarning, match="Using 'legacy' keyword is deprecated"):
                result = special.comb(N, k, exact=True, legacy=legacy, repetition=repetition)
        else:
            result = special.comb(N, k, exact=True, legacy=legacy, repetition=repetition)
        if legacy:
            if repetition:
                N, k = (int(N + k - 1), int(k))
                repetition = False
            else:
                N, k = (int(N), int(k))
        with suppress_warnings() as sup:
            if legacy is not _NoValue:
                sup.filter(DeprecationWarning)
            expected = special.comb(N, k, legacy=legacy, repetition=repetition)
        assert_equal(result, expected)

    def test_comb_with_np_int64(self):
        n = 70
        k = 30
        np_n = np.int64(n)
        np_k = np.int64(k)
        res_np = special.comb(np_n, np_k, exact=True)
        res_py = special.comb(n, k, exact=True)
        assert res_np == res_py

    def test_comb_zeros(self):
        assert_equal(special.comb(2, 3, exact=True), 0)
        assert_equal(special.comb(-1, 3, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=True), 0)
        assert_equal(special.comb(2, -1, exact=False), 0)
        assert_array_almost_equal(special.comb([2, -1, 2, 10], [3, 3, -1, 3]), [0.0, 0.0, 0.0, 120.0])

    def test_perm(self):
        assert_array_almost_equal(special.perm([10, 10], [3, 4]), [720.0, 5040.0])
        assert_almost_equal(special.perm(10, 3), 720.0)
        assert_equal(special.perm(10, 3, exact=True), 720)

    def test_perm_zeros(self):
        assert_equal(special.perm(2, 3, exact=True), 0)
        assert_equal(special.perm(-1, 3, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=True), 0)
        assert_equal(special.perm(2, -1, exact=False), 0)
        assert_array_almost_equal(special.perm([2, -1, 2, 10], [3, 3, -1, 3]), [0.0, 0.0, 0.0, 720.0])

    def test_positional_deprecation(self):
        with pytest.deprecated_call(match='use keyword arguments'):
            special.comb([10, 10], [3, 4], False, False)