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
class TestRemainder:

    def test_remainder_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)
                    b = np.array(sg2 * 19, dtype=dt2)
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_remainder_exact(self):
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = list((divmod(*t) for t in arg))
        a, b = np.array(arg, dtype=int).T
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                fa = a.astype(dt)
                fb = b.astype(dt)
                div, rem = op(fa, fb)
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_remainder_roundoff(self):
        dt = np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-08, dtype=dt1)
                    b = np.array(sg2 * 6e-08, dtype=dt2)
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith('darwin'), reason="MacOS seems to not give the correct 'invalid' warning for `fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_float_divmod_errors(self, dtype):
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fzero, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, finf)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, fzero)
        with np.errstate(divide='raise', invalid='ignore'):
            np.divmod(finf, fzero)

    @pytest.mark.skipif(hasattr(np.__config__, 'blas_ssl2_info'), reason='gh-22982')
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith('darwin'), reason="MacOS seems to not give the correct 'invalid' warning for `fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    @pytest.mark.parametrize('fn', [np.fmod, np.remainder])
    def test_float_remainder_errors(self, dtype, fn):
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        with np.errstate(all='raise'):
            with pytest.raises(FloatingPointError, match='invalid value'):
                fn(fone, fzero)
            fn(fnan, fzero)
            fn(fzero, fnan)
            fn(fone, fnan)
            fn(fnan, fone)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_float_remainder_overflow(self):
        a = np.finfo(np.float64).tiny
        with np.errstate(over='ignore', invalid='ignore'):
            div, mod = np.divmod(4, a)
            np.isinf(div)
            assert_(mod == 0)
        with np.errstate(over='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)
        with np.errstate(invalid='raise', over='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)

    def test_float_divmod_corner_cases(self):
        for dt in np.typecodes['Float']:
            fnan = np.array(np.nan, dtype=dt)
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'invalid value encountered in divmod')
                sup.filter(RuntimeWarning, 'divide by zero encountered in divmod')
                div, rem = np.divmod(fone, fzer)
                assert np.isinf(div), 'dt: %s, div: %s' % (dt, rem)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fzer, fzer)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                (assert_(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem))
                div, rem = np.divmod(finf, finf)
                assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(finf, fzer)
                assert np.isinf(div), 'dt: %s, rem: %s' % (dt, rem)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fnan, fone)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fone, fnan)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fnan, fzer)
                assert np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem)
                assert np.isnan(div), 'dt: %s, rem: %s' % (dt, rem)

    def test_float_remainder_corner_cases(self):
        for dt in np.typecodes['Float']:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = np.remainder(a, b)
            assert_(rem <= b, 'dt: %s' % dt)
            rem = np.remainder(-a, -b)
            assert_(rem >= -b, 'dt: %s' % dt)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in remainder')
            sup.filter(RuntimeWarning, 'invalid value encountered in fmod')
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = np.remainder(fone, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                rem = np.remainder(finf, fone)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                rem = np.remainder(finf, finf)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(finf, fzer)
                fmod = np.fmod(finf, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(fone, fnan)
                fmod = np.fmod(fone, fnan)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(fnan, fzer)
                fmod = np.fmod(fnan, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
                rem = np.remainder(fnan, fone)
                fmod = np.fmod(fnan, fone)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))