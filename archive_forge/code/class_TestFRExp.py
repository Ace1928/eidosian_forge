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
class TestFRExp:

    @pytest.mark.parametrize('stride', [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize('dtype', ['f', 'd'])
    @pytest.mark.xfail(IS_MUSL, reason='gh23048')
    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason='np.frexp gives different answers for NAN/INF on windows and linux')
    def test_frexp(self, dtype, stride):
        arr = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
        mant_true = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 0.5, -0.5], dtype=dtype)
        exp_true = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype='i')
        out_mant = np.ones(8, dtype=dtype)
        out_exp = 2 * np.ones(8, dtype='i')
        mant, exp = np.frexp(arr[::stride], out=(out_mant[::stride], out_exp[::stride]))
        assert_equal(mant_true[::stride], mant)
        assert_equal(exp_true[::stride], exp)
        assert_equal(out_mant[::stride], mant_true[::stride])
        assert_equal(out_exp[::stride], exp_true[::stride])