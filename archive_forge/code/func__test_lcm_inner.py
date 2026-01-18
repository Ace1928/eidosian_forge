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
def _test_lcm_inner(self, dtype):
    a = np.array([12, 120], dtype=dtype)
    b = np.array([20, 200], dtype=dtype)
    assert_equal(np.lcm(a, b), [60, 600])
    if not issubclass(dtype, np.unsignedinteger):
        a = np.array([12, -12, 12, -12], dtype=dtype)
        b = np.array([20, 20, -20, -20], dtype=dtype)
        assert_equal(np.lcm(a, b), [60] * 4)
    a = np.array([3, 12, 20], dtype=dtype)
    assert_equal(np.lcm.reduce([3, 12, 20]), 60)
    a = np.arange(6).astype(dtype)
    b = 20
    assert_equal(np.lcm(a, b), [0, 20, 20, 60, 20, 20])