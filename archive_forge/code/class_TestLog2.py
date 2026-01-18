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
class TestLog2:

    @pytest.mark.parametrize('dt', ['f', 'd', 'g'])
    def test_log2_values(self, dt):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        assert_almost_equal(np.log2(xf), yf)

    @pytest.mark.parametrize('i', range(1, 65))
    def test_log2_ints(self, i):
        v = np.log2(2.0 ** i)
        assert_equal(v, float(i), err_msg='at exponent %d' % i)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log2_special(self):
        assert_equal(np.log2(1.0), 0.0)
        assert_equal(np.log2(np.inf), np.inf)
        assert_(np.isnan(np.log2(np.nan)))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.log2(-1.0)))
            assert_(np.isnan(np.log2(-np.inf)))
            assert_equal(np.log2(0.0), -np.inf)
            assert_(w[0].category is RuntimeWarning)
            assert_(w[1].category is RuntimeWarning)
            assert_(w[2].category is RuntimeWarning)