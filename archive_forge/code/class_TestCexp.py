import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestCexp:

    def test_simple(self):
        check = check_complex_value
        f = np.exp
        check(f, 1, 0, np.exp(1), 0, False)
        check(f, 0, 1, np.cos(1), np.sin(1), False)
        ref = np.exp(1) * complex(np.cos(1), np.sin(1))
        check(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        check = check_complex_value
        f = np.exp
        check(f, np.PZERO, 0, 1, 0, False)
        check(f, np.NZERO, 0, 1, 0, False)
        check(f, 1, np.inf, np.nan, np.nan)
        check(f, -1, np.inf, np.nan, np.nan)
        check(f, 0, np.inf, np.nan, np.nan)
        check(f, np.inf, 0, np.inf, 0)
        check(f, -np.inf, 1, np.PZERO, np.PZERO)
        check(f, -np.inf, 0.75 * np.pi, np.NZERO, np.PZERO)
        check(f, np.inf, 1, np.inf, np.inf)
        check(f, np.inf, 0.75 * np.pi, -np.inf, np.inf)

        def _check_ninf_inf(dummy):
            msgform = 'cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.inf)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_inf(None)

        def _check_inf_inf(dummy):
            msgform = 'cexp(inf, inf) is (%f, %f), expected (+-inf, nan)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.inf)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_inf_inf(None)

        def _check_ninf_nan(dummy):
            msgform = 'cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.nan)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_ninf_nan(None)

        def _check_inf_nan(dummy):
            msgform = 'cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)'
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.nan)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))
        _check_inf_nan(None)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, -1, np.nan, np.nan)
        check(f, np.nan, np.inf, np.nan, np.nan)
        check(f, np.nan, -np.inf, np.nan, np.nan)
        check(f, np.nan, np.nan, np.nan, np.nan)

    @pytest.mark.skip(reason='cexp(nan + 0I) is wrong on most platforms')
    def test_special_values2(self):
        check = check_complex_value
        f = np.exp
        check(f, np.nan, 0, np.nan, 0)