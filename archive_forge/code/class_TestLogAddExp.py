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
class TestLogAddExp(_FilterInvalids):

    def test_logaddexp_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            xf = np.log(np.array(x, dtype=dt))
            yf = np.log(np.array(y, dtype=dt))
            zf = np.log(np.array(z, dtype=dt))
            assert_almost_equal(np.logaddexp(xf, yf), zf, decimal=dec_)

    def test_logaddexp_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for dt in ['f', 'd', 'g']:
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            assert_almost_equal(np.logaddexp(logxf, logyf), logzf)

    def test_inf(self):
        inf = np.inf
        x = [inf, -inf, inf, -inf, inf, 1, -inf, 1]
        y = [inf, inf, -inf, -inf, 1, inf, 1, -inf]
        z = [inf, inf, inf, -inf, inf, inf, 1, 1]
        with np.errstate(invalid='raise'):
            for dt in ['f', 'd', 'g']:
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                assert_equal(np.logaddexp(logxf, logyf), logzf)

    def test_nan(self):
        assert_(np.isnan(np.logaddexp(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, 0)))
        assert_(np.isnan(np.logaddexp(0, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, np.nan)))

    def test_reduce(self):
        assert_equal(np.logaddexp.identity, -np.inf)
        assert_equal(np.logaddexp.reduce([]), -np.inf)