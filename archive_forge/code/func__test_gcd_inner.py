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
def _test_gcd_inner(self, dtype):
    a = np.array([12, 120], dtype=dtype)
    b = np.array([20, 200], dtype=dtype)
    assert_equal(np.gcd(a, b), [4, 40])
    if not issubclass(dtype, np.unsignedinteger):
        a = np.array([12, -12, 12, -12], dtype=dtype)
        b = np.array([20, 20, -20, -20], dtype=dtype)
        assert_equal(np.gcd(a, b), [4] * 4)
    a = np.array([15, 25, 35], dtype=dtype)
    assert_equal(np.gcd.reduce(a), 5)
    a = np.arange(6).astype(dtype)
    b = 20
    assert_equal(np.gcd(a, b), [20, 1, 2, 1, 4, 5])