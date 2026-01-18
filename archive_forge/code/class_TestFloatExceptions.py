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
class TestFloatExceptions:

    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False, "Type %s did not raise fpe error '%s'." % (ftype, fpeerr))
        except FloatingPointError as exc:
            assert_(str(exc).find(fpeerr) >= 0, "Type %s raised wrong fpe error '%s'." % (ftype, exc))

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    @pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
    @pytest.mark.parametrize('typecode', np.typecodes['AllFloat'])
    def test_floating_exceptions(self, typecode):
        if 'bsd' in sys.platform and typecode in 'gG':
            pytest.skip(reason='Fallback impl for (c)longdouble may not raise FPE errors as expected on BSD OSes, see gh-24876, gh-23379')
        with np.errstate(all='raise'):
            ftype = np.obj2sctype(typecode)
            if np.dtype(ftype).kind == 'f':
                fi = np.finfo(ftype)
                ft_tiny = fi._machar.tiny
                ft_max = fi.max
                ft_eps = fi.eps
                underflow = 'underflow'
                divbyzero = 'divide by zero'
            else:
                rtype = type(ftype(0).real)
                fi = np.finfo(rtype)
                ft_tiny = ftype(fi._machar.tiny)
                ft_max = ftype(fi.max)
                ft_eps = ftype(fi.eps)
                underflow = ''
                divbyzero = ''
            overflow = 'overflow'
            invalid = 'invalid'
            if not np.isnan(ft_tiny):
                self.assert_raises_fpe(underflow, lambda a, b: a / b, ft_tiny, ft_max)
                self.assert_raises_fpe(underflow, lambda a, b: a * b, ft_tiny, ft_tiny)
            self.assert_raises_fpe(overflow, lambda a, b: a * b, ft_max, ftype(2))
            self.assert_raises_fpe(overflow, lambda a, b: a / b, ft_max, ftype(0.5))
            self.assert_raises_fpe(overflow, lambda a, b: a + b, ft_max, ft_max * ft_eps)
            self.assert_raises_fpe(overflow, lambda a, b: a - b, -ft_max, ft_max * ft_eps)
            self.assert_raises_fpe(overflow, np.power, ftype(2), ftype(2 ** fi.nexp))
            self.assert_raises_fpe(divbyzero, lambda a, b: a / b, ftype(1), ftype(0))
            self.assert_raises_fpe(invalid, lambda a, b: a / b, ftype(np.inf), ftype(np.inf))
            self.assert_raises_fpe(invalid, lambda a, b: a / b, ftype(0), ftype(0))
            self.assert_raises_fpe(invalid, lambda a, b: a - b, ftype(np.inf), ftype(np.inf))
            self.assert_raises_fpe(invalid, lambda a, b: a + b, ftype(np.inf), ftype(-np.inf))
            self.assert_raises_fpe(invalid, lambda a, b: a * b, ftype(0), ftype(np.inf))

    @pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with np.errstate(all='warn'):
                np.divide(1, 0.0)
                assert_equal(len(w), 1)
                assert_('divide by zero' in str(w[0].message))
                np.array(1e+300) * np.array(1e+300)
                assert_equal(len(w), 2)
                assert_('overflow' in str(w[-1].message))
                np.array(np.inf) - np.array(np.inf)
                assert_equal(len(w), 3)
                assert_('invalid value' in str(w[-1].message))
                np.array(1e-300) * np.array(1e-300)
                assert_equal(len(w), 4)
                assert_('underflow' in str(w[-1].message))