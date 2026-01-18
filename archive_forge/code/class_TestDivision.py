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
class TestDivision:

    def test_division_int(self):
        x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        if 5 / 10 == 0.5:
            assert_equal(x / 100, [0.05, 0.1, 0.9, 1, -0.05, -0.1, -0.9, -1, -1.2])
        else:
            assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype,ex_val', itertools.product(np.sctypes['int'] + np.sctypes['uint'], ('np.array(range(fo.max-lsize, fo.max)).astype(dtype),np.arange(lsize).astype(dtype),range(15)', 'np.arange(fo.min, fo.min+lsize).astype(dtype),np.arange(lsize//-2, lsize//2).astype(dtype),range(fo.min, fo.min + 15)', 'np.array(range(fo.max-lsize, fo.max)).astype(dtype),np.arange(lsize).astype(dtype),[1,3,9,13,neg, fo.min+1, fo.min//2, fo.max//3, fo.max//4]')))
    def test_division_int_boundary(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        neg = -1 if fo.min < 0 else 1
        lsize = 512 + 7
        a, b, divisors = eval(ex_val)
        a_lst, b_lst = (a.tolist(), b.tolist())
        c_div = lambda n, d: 0 if d == 0 else fo.min if n and n == fo.min and (d == -1) else n // d
        with np.errstate(divide='ignore'):
            ac = a.copy()
            ac //= b
            div_ab = a // b
        div_lst = [c_div(x, y) for x, y in zip(a_lst, b_lst)]
        msg = 'Integer arrays floor division check (//)'
        assert all(div_ab == div_lst), msg
        msg_eq = 'Integer arrays floor division check (//=)'
        assert all(ac == div_lst), msg_eq
        for divisor in divisors:
            ac = a.copy()
            with np.errstate(divide='ignore', over='ignore'):
                div_a = a // divisor
                ac //= divisor
            div_lst = [c_div(i, divisor) for i in a_lst]
            assert all(div_a == div_lst), msg
            assert all(ac == div_lst), msg_eq
        with np.errstate(divide='raise', over='raise'):
            if 0 in b:
                with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
                    a // b
            else:
                a // b
            if fo.min and fo.min in a:
                with pytest.raises(FloatingPointError, match='overflow encountered in floor_divide'):
                    a // -1
            elif fo.min:
                a // -1
            with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
                a // 0
            with pytest.raises(FloatingPointError, match='divide by zero encountered in floor_divide'):
                ac = a.copy()
                ac //= 0
            np.array([], dtype=dtype) // 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype,ex_val', itertools.product(np.sctypes['int'] + np.sctypes['uint'], ('np.array([fo.max, 1, 2, 1, 1, 2, 3], dtype=dtype)', 'np.array([fo.min, 1, -2, 1, 1, 2, -3]).astype(dtype)', 'np.arange(fo.min, fo.min+(100*10), 10, dtype=dtype)', 'np.array(range(fo.max-(100*7), fo.max, 7)).astype(dtype)')))
    def test_division_int_reduce(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        a = eval(ex_val)
        lst = a.tolist()
        c_div = lambda n, d: 0 if d == 0 or (n and n == fo.min and (d == -1)) else n // d
        with np.errstate(divide='ignore'):
            div_a = np.floor_divide.reduce(a)
        div_lst = reduce(c_div, lst)
        msg = 'Reduce floor integer division check'
        assert div_a == div_lst, msg
        with np.errstate(divide='raise', over='raise'):
            with pytest.raises(FloatingPointError, match='divide by zero encountered in reduce'):
                np.floor_divide.reduce(np.arange(-100, 100).astype(dtype))
            if fo.min:
                with pytest.raises(FloatingPointError, match='overflow encountered in reduce'):
                    np.floor_divide.reduce(np.array([fo.min, 1, -1], dtype=dtype))

    @pytest.mark.parametrize('dividend,divisor,quotient', [(np.timedelta64(2, 'Y'), np.timedelta64(2, 'M'), 12), (np.timedelta64(2, 'Y'), np.timedelta64(-2, 'M'), -12), (np.timedelta64(-2, 'Y'), np.timedelta64(2, 'M'), -12), (np.timedelta64(-2, 'Y'), np.timedelta64(-2, 'M'), 12), (np.timedelta64(2, 'M'), np.timedelta64(-2, 'Y'), -1), (np.timedelta64(2, 'Y'), np.timedelta64(0, 'M'), 0), (np.timedelta64(2, 'Y'), 2, np.timedelta64(1, 'Y')), (np.timedelta64(2, 'Y'), -2, np.timedelta64(-1, 'Y')), (np.timedelta64(-2, 'Y'), 2, np.timedelta64(-1, 'Y')), (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')), (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')), (np.timedelta64(-2, 'Y'), -3, np.timedelta64(0, 'Y')), (np.timedelta64(-2, 'Y'), 0, np.timedelta64('Nat', 'Y'))])
    def test_division_int_timedelta(self, dividend, divisor, quotient):
        if divisor and (isinstance(quotient, int) or not np.isnat(quotient)):
            msg = 'Timedelta floor division check'
            assert dividend // divisor == quotient, msg
            msg = 'Timedelta arrays floor division check'
            dividend_array = np.array([dividend] * 5)
            quotient_array = np.array([quotient] * 5)
            assert all(dividend_array // divisor == quotient_array), msg
        else:
            if IS_WASM:
                pytest.skip("fp errors don't work in wasm")
            with np.errstate(divide='raise', invalid='raise'):
                with pytest.raises(FloatingPointError):
                    dividend // divisor

    def test_division_complex(self):
        msg = 'Complex division implementation check'
        x = np.array([1.0 + 1.0 * 1j, 1.0 + 0.5 * 1j, 1.0 + 2.0 * 1j], dtype=np.complex128)
        assert_almost_equal(x ** 2 / x, x, err_msg=msg)
        msg = 'Complex division overflow/underflow check'
        x = np.array([1e+110, 1e-110], dtype=np.complex128)
        y = x ** 2 / x
        assert_almost_equal(y / x, [1, 1], err_msg=msg)

    def test_zero_division_complex(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.array([0.0], dtype=np.complex128)
            y = 1.0 / x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.nan) / x
            assert_(np.isinf(y)[0])
            y = complex(np.nan, np.inf) / x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.inf) / x
            assert_(np.isinf(y)[0])
            y = 0.0 / x
            assert_(np.isnan(y)[0])

    def test_floor_division_complex(self):
        x = np.array([0.9 + 1j, -0.1 + 1j, 0.9 + 0.5 * 1j, 0.9 + 2.0 * 1j], dtype=np.complex128)
        with pytest.raises(TypeError):
            x // 7
        with pytest.raises(TypeError):
            np.divmod(x, 7)
        with pytest.raises(TypeError):
            np.remainder(x, 7)

    def test_floor_division_signed_zero(self):
        x = np.zeros(10)
        assert_equal(np.signbit(x // 1), 0)
        assert_equal(np.signbit(-x // 1), 1)

    @pytest.mark.skipif(hasattr(np.__config__, 'blas_ssl2_info'), reason='gh-22982')
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_errors(self, dtype):
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.floor_divide, fone, fzer)
        with np.errstate(divide='ignore', invalid='raise'):
            np.floor_divide(fone, fzer)
        with np.errstate(all='raise'):
            np.floor_divide(fnan, fone)
            np.floor_divide(fone, fnan)
            np.floor_divide(fnan, fzer)
            np.floor_divide(fzer, fnan)

    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_corner_cases(self, dtype):
        x = np.zeros(10, dtype=dtype)
        y = np.ones(10, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in floor_divide')
            div = np.floor_divide(fnan, fone)
            assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
            div = np.floor_divide(fone, fnan)
            assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
            div = np.floor_divide(fnan, fzer)
            assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
        with np.errstate(divide='ignore'):
            z = np.floor_divide(y, x)
            assert_(np.isinf(z).all())