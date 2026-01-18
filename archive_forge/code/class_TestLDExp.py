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
class TestLDExp:

    @pytest.mark.parametrize('stride', [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize('dtype', ['f', 'd'])
    def test_ldexp(self, dtype, stride):
        mant = np.array([0.125, 0.25, 0.5, 1.0, 1.0, 2.0, 4.0, 8.0], dtype=dtype)
        exp = np.array([3, 2, 1, 0, 0, -1, -2, -3], dtype='i')
        out = np.zeros(8, dtype=dtype)
        assert_equal(np.ldexp(mant[::stride], exp[::stride], out=out[::stride]), np.ones(8, dtype=dtype)[::stride])
        assert_equal(out[::stride], np.ones(8, dtype=dtype)[::stride])