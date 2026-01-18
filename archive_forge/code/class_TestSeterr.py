import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestSeterr:

    def test_default(self):
        err = np.geterr()
        assert_equal(err, dict(divide='warn', invalid='warn', over='warn', under='ignore'))

    def test_set(self):
        with np.errstate():
            err = np.seterr()
            old = np.seterr(divide='print')
            assert_(err == old)
            new = np.seterr()
            assert_(new['divide'] == 'print')
            np.seterr(over='raise')
            assert_(np.geterr()['over'] == 'raise')
            assert_(new['divide'] == 'print')
            np.seterr(**old)
            assert_(np.geterr() == old)

    @pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
    @pytest.mark.skipif(platform.machine() == 'armv5tel', reason='See gh-413.')
    def test_divide_err(self):
        with np.errstate(divide='raise'):
            with assert_raises(FloatingPointError):
                np.array([1.0]) / np.array([0.0])
            np.seterr(divide='ignore')
            np.array([1.0]) / np.array([0.0])

    @pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
    def test_errobj(self):
        olderrobj = np.geterrobj()
        self.called = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                with np.errstate(divide='warn'):
                    np.seterrobj([20000, 1, None])
                    np.array([1.0]) / np.array([0.0])
                    assert_equal(len(w), 1)

            def log_err(*args):
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_('divide' in extobj_err[0])
            with np.errstate(divide='ignore'):
                np.seterrobj([20000, 3, log_err])
                np.array([1.0]) / np.array([0.0])
            assert_equal(self.called, 1)
            np.seterrobj(olderrobj)
            with np.errstate(divide='ignore'):
                np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
            assert_equal(self.called, 2)
        finally:
            np.seterrobj(olderrobj)
            del self.called

    def test_errobj_noerrmask(self):
        olderrobj = np.geterrobj()
        try:
            np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT, umath.ERR_DEFAULT + 1, None])
            np.isnan(np.array([6]))
            for i in range(10000):
                np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT, umath.ERR_DEFAULT, None])
            np.isnan(np.array([6]))
        finally:
            np.seterrobj(olderrobj)