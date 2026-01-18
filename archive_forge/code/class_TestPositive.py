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
class TestPositive:

    def test_valid(self):
        valid_dtypes = [int, float, complex, object]
        for dtype in valid_dtypes:
            x = np.arange(5, dtype=dtype)
            result = np.positive(x)
            assert_equal(x, result, err_msg=str(dtype))

    def test_invalid(self):
        with assert_raises(TypeError):
            np.positive(True)
        with assert_raises(TypeError):
            np.positive(np.datetime64('2000-01-01'))
        with assert_raises(TypeError):
            np.positive(np.array(['foo'], dtype=str))
        with assert_raises(TypeError):
            np.positive(np.array(['bar'], dtype=object))