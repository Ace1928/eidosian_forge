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
class TestErf:

    def test_erf(self):
        er = special.erf(0.25)
        assert_almost_equal(er, 0.2763263902, 8)

    def test_erf_zeros(self):
        erz = special.erf_zeros(5)
        erzr = array([1.45061616 + 1.880943j, 2.24465928 + 2.61657514j, 2.83974105 + 3.1756281j, 3.33546074 + 3.64617438j, 3.76900557 + 4.06069723j])
        assert_array_almost_equal(erz, erzr, 4)

    def _check_variant_func(self, func, other_func, rtol, atol=0):
        np.random.seed(1234)
        n = 10000
        x = np.random.pareto(0.02, n) * (2 * np.random.randint(0, 2, n) - 1)
        y = np.random.pareto(0.02, n) * (2 * np.random.randint(0, 2, n) - 1)
        z = x + 1j * y
        with np.errstate(all='ignore'):
            w = other_func(z)
            w_real = other_func(x).real
            mask = np.isfinite(w)
            w = w[mask]
            z = z[mask]
            mask = np.isfinite(w_real)
            w_real = w_real[mask]
            x = x[mask]
            assert_func_equal(func, w, z, rtol=rtol, atol=atol)
            assert_func_equal(func, w_real, x, rtol=rtol, atol=atol)

    def test_erfc_consistent(self):
        self._check_variant_func(cephes.erfc, lambda z: 1 - cephes.erf(z), rtol=1e-12, atol=1e-14)

    def test_erfcx_consistent(self):
        self._check_variant_func(cephes.erfcx, lambda z: np.exp(z * z) * cephes.erfc(z), rtol=1e-12)

    def test_erfi_consistent(self):
        self._check_variant_func(cephes.erfi, lambda z: -1j * cephes.erf(1j * z), rtol=1e-12)

    def test_dawsn_consistent(self):
        self._check_variant_func(cephes.dawsn, lambda z: sqrt(pi) / 2 * np.exp(-z * z) * cephes.erfi(z), rtol=1e-12)

    def test_erf_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -1, 1]
        assert_allclose(special.erf(vals), expected, rtol=1e-15)

    def test_erfc_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, 2, 0]
        assert_allclose(special.erfc(vals), expected, rtol=1e-15)

    def test_erfcx_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, np.inf, 0]
        assert_allclose(special.erfcx(vals), expected, rtol=1e-15)

    def test_erfi_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -np.inf, np.inf]
        assert_allclose(special.erfi(vals), expected, rtol=1e-15)

    def test_dawsn_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan, -0.0, 0.0]
        assert_allclose(special.dawsn(vals), expected, rtol=1e-15)

    def test_wofz_nan_inf(self):
        vals = [np.nan, -np.inf, np.inf]
        expected = [np.nan + np.nan * 1j, 0.0 - 0j, 0.0 + 0j]
        assert_allclose(special.wofz(vals), expected, rtol=1e-15)