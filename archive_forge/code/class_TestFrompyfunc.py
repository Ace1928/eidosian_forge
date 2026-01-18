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
class TestFrompyfunc:

    def test_identity(self):

        def mul(a, b):
            return a * b
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_equal(mul_ufunc.reduce([]), 1)
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=None)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_raises(ValueError, lambda: mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)))
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))