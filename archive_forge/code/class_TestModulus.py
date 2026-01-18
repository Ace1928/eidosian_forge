import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
class TestModulus:

    def test_modulus_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)[()]
                    b = np.array(sg2 * 19, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_exact(self):
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
        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                fa = a.astype(dt)
                fb = b.astype(dt)
                div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_modulus_roundoff(self):
        dt = np.typecodes['Float']
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-08, dtype=dt1)[()]
                    b = np.array(sg2 * 6e-08, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_modulus_corner_cases(self):
        for dt in np.typecodes['Float']:
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = operator.mod(a, b)
            assert_(rem <= b, 'dt: %s' % dt)
            rem = operator.mod(-a, -b)
            assert_(rem >= -b, 'dt: %s' % dt)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in remainder')
            sup.filter(RuntimeWarning, 'divide by zero encountered in remainder')
            sup.filter(RuntimeWarning, 'divide by zero encountered in floor_divide')
            sup.filter(RuntimeWarning, 'divide by zero encountered in divmod')
            sup.filter(RuntimeWarning, 'invalid value encountered in divmod')
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = operator.mod(fone, fzer)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                rem = operator.mod(fone, fnan)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                rem = operator.mod(finf, fone)
                assert_(np.isnan(rem), 'dt: %s' % dt)
                for op in [floordiv_and_mod, divmod]:
                    div, mod = op(fone, fzer)
                    assert_(np.isinf(div)) and assert_(np.isnan(mod))

    def test_inplace_floordiv_handling(self):
        a = np.array([1, 2], np.int64)
        b = np.array([1, 2], np.uint64)
        with pytest.raises(TypeError, match="Cannot cast ufunc 'floor_divide' output from"):
            a //= b