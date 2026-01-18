import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestSpecialFloats:

    def test_exp_values(self):
        with np.errstate(under='raise', over='raise'):
            x = [np.nan, np.nan, np.inf, 0.0]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.exp(yf), xf)

    @pytest.mark.xfail(_glibc_older_than('2.17'), reason='Older glibc versions may not raise appropriate FP exceptions')
    def test_exp_exceptions(self):
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(11.0899))
            assert_raises(FloatingPointError, np.exp, np.float32(100.0))
            assert_raises(FloatingPointError, np.exp, np.float32(1e+19))
            assert_raises(FloatingPointError, np.exp, np.float64(800.0))
            assert_raises(FloatingPointError, np.exp, np.float64(1e+19))
        with np.errstate(under='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(-17.5))
            assert_raises(FloatingPointError, np.exp, np.float32(-1000.0))
            assert_raises(FloatingPointError, np.exp, np.float32(-1e+19))
            assert_raises(FloatingPointError, np.exp, np.float64(-1000.0))
            assert_raises(FloatingPointError, np.exp, np.float64(-1e+19))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, -np.inf, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -1.0]
            y1p = [np.nan, -np.nan, np.inf, -np.inf, -1.0, -2.0]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                yf1p = np.array(y1p, dtype=dt)
                assert_equal(np.log(yf), xf)
                assert_equal(np.log2(yf), xf)
                assert_equal(np.log10(yf), xf)
                assert_equal(np.log1p(yf1p), xf)
        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log, np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2, np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10, np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p, np.array(-1.0, dtype=dt))
        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log, np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log, np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2, np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log2, np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10, np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log10, np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p, np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log1p, np.array(-2.0, dtype=dt))
        with assert_no_warnings():
            a = np.array(1000000000.0, dtype='float32')
            np.log(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', ['e', 'f', 'd', 'g'])
    def test_sincos_values(self, dtype):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.nan, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            xf = np.array(x, dtype=dtype)
            yf = np.array(y, dtype=dtype)
            assert_equal(np.sin(yf), xf)
            assert_equal(np.cos(yf), xf)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith('darwin'), reason="underflow is triggered for scalar 'sin'")
    def test_sincos_underflow(self):
        with np.errstate(under='raise'):
            underflow_trigger = np.array(float.fromhex('0x1.f37f47a03f82ap-511'), dtype=np.float64)
            np.sin(underflow_trigger)
            np.cos(underflow_trigger)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['e', 'f', 'd'])
    @pytest.mark.parametrize('value', [np.inf, -np.inf])
    def test_sincos_errors(self, callable, dtype, value):
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, callable, np.array([value], dtype=dtype))

    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['f', 'd'])
    @pytest.mark.parametrize('stride', [-1, 1, 2, 4, 5])
    def test_sincos_overlaps(self, callable, dtype, stride):
        N = 100
        M = N // abs(stride)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N, dtype)
        y = callable(x[::stride])
        callable(x[::stride], out=x[:M])
        assert_equal(x[:M], y)

    @pytest.mark.parametrize('dt', ['e', 'f', 'd', 'g'])
    def test_sqrt_values(self, dt):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, 0.0]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0]
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.sqrt(yf), xf)

    def test_abs_values(self):
        x = [np.nan, np.nan, np.inf, np.inf, 0.0, 0.0, 1.0, 1.0]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, -1.0, 1.0]
        for dt in ['e', 'f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.abs(yf), xf)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_square_values(self):
        x = [np.nan, np.nan, np.inf, np.inf]
        y = [np.nan, -np.nan, np.inf, -np.inf]
        with np.errstate(all='ignore'):
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.square(yf), xf)
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.square, np.array(1000.0, dtype='e'))
            assert_raises(FloatingPointError, np.square, np.array(1e+32, dtype='f'))
            assert_raises(FloatingPointError, np.square, np.array(1e+200, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_reciprocal_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, 0.0, -0.0, np.inf, -np.inf]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.reciprocal(yf), xf)
        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd', 'g']:
                assert_raises(FloatingPointError, np.reciprocal, np.array(-0.0, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_tan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf]
            out = [np.nan, np.nan, 0.0, -0.0, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.tan(in_arr), out_arr)
        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.tan, np.array(np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.tan, np.array(-np.inf, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arcsincos(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arcsin(in_arr), out_arr)
                assert_equal(np.arccos(in_arr), out_arr)
        for callable in [np.arcsin, np.arccos]:
            for value in [np.inf, -np.inf, 2.0, -2.0]:
                for dt in ['e', 'f', 'd']:
                    with np.errstate(invalid='raise'):
                        assert_raises(FloatingPointError, callable, np.array(value, dtype=dt))

    def test_arctan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan]
            out = [np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctan(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.sinh(in_arr), out_arr)
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.sinh, np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.sinh, np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.sinh, np.array(1200.0, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif('bsd' in sys.platform, reason='fallback implementation may not raise, see gh-2487')
    def test_cosh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.cosh(in_arr), out_arr)
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.cosh, np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.cosh, np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.cosh, np.array(1200.0, dtype='d'))

    def test_tanh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, 1.0, -1.0]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.tanh(in_arr), out_arr)

    def test_arcsinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.arcsinh(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arccosh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, 0.0]
            out = [np.nan, np.nan, np.inf, np.nan, 0.0, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arccosh(in_arr), out_arr)
        for value in [0.0, -np.inf]:
            with np.errstate(invalid='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arccosh, np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arctanh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, -1.0, 2.0]
            out = [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctanh(in_arr), out_arr)
        for value in [1.01, np.inf, -np.inf, 1.0, -1.0]:
            with np.errstate(invalid='raise', divide='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arctanh, np.array(value, dtype=dt))
        assert np.signbit(np.arctanh(-1j).real)

    @pytest.mark.xfail(_glibc_older_than('2.17'), reason='Older glibc versions may not raise appropriate FP exceptions')
    def test_exp2(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, 0.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.exp2(in_arr), out_arr)
        for value in [2000.0, -2000.0]:
            with np.errstate(over='raise', under='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.exp2, np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_expm1(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, -1.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.expm1(in_arr), out_arr)
        for value in [200.0, 2000.0]:
            with np.errstate(over='raise'):
                for dt in ['e', 'f']:
                    assert_raises(FloatingPointError, np.expm1, np.array(value, dtype=dt))
    INF_INVALID_ERR = [np.cos, np.sin, np.tan, np.arccos, np.arcsin, np.spacing, np.arctanh]
    NEG_INVALID_ERR = [np.log, np.log2, np.log10, np.log1p, np.sqrt, np.arccosh, np.arctanh]
    ONE_INVALID_ERR = [np.arctanh]
    LTONE_INVALID_ERR = [np.arccosh]
    BYZERO_ERR = [np.log, np.log2, np.log10, np.reciprocal, np.arccosh]

    @pytest.mark.skipif(sys.platform == 'win32' and sys.maxsize < 2 ** 31 + 1, reason='failures on 32-bit Python, see FIXME below')
    @pytest.mark.parametrize('ufunc', UFUNCS_UNARY_FP)
    @pytest.mark.parametrize('dtype', ('e', 'f', 'd'))
    @pytest.mark.parametrize('data, escape', (([0.03], LTONE_INVALID_ERR), ([0.03] * 32, LTONE_INVALID_ERR), ([-1.0], NEG_INVALID_ERR), ([-1.0] * 32, NEG_INVALID_ERR), ([1.0], ONE_INVALID_ERR), ([1.0] * 32, ONE_INVALID_ERR), ([0.0], BYZERO_ERR), ([0.0] * 32, BYZERO_ERR), ([-0.0], BYZERO_ERR), ([-0.0] * 32, BYZERO_ERR), ([0.5, 0.5, 0.5, np.nan], LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, np.nan] * 32, LTONE_INVALID_ERR), ([np.nan, 1.0, 1.0, 1.0], ONE_INVALID_ERR), ([np.nan, 1.0, 1.0, 1.0] * 32, ONE_INVALID_ERR), ([np.nan], []), ([np.nan] * 32, []), ([0.5, 0.5, 0.5, np.inf], INF_INVALID_ERR + LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, np.inf] * 32, INF_INVALID_ERR + LTONE_INVALID_ERR), ([np.inf, 1.0, 1.0, 1.0], INF_INVALID_ERR), ([np.inf, 1.0, 1.0, 1.0] * 32, INF_INVALID_ERR), ([np.inf], INF_INVALID_ERR), ([np.inf] * 32, INF_INVALID_ERR), ([0.5, 0.5, 0.5, -np.inf], NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, -np.inf] * 32, NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR), ([-np.inf, 1.0, 1.0, 1.0], NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf, 1.0, 1.0, 1.0] * 32, NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf], NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf] * 32, NEG_INVALID_ERR + INF_INVALID_ERR)))
    def test_unary_spurious_fpexception(self, ufunc, dtype, data, escape):
        if escape and ufunc in escape:
            return
        if ufunc in (np.spacing, np.ceil) and dtype == 'e':
            return
        array = np.array(data, dtype=dtype)
        with assert_no_warnings():
            ufunc(array)

    @pytest.mark.parametrize('dtype', ('e', 'f', 'd'))
    def test_divide_spurious_fpexception(self, dtype):
        dt = np.dtype(dtype)
        dt_info = np.finfo(dt)
        subnorm = dt_info.smallest_subnormal
        with assert_no_warnings():
            np.zeros(128 + 1, dtype=dt) / subnorm