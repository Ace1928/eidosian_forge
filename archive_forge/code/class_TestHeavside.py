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
class TestHeavside:

    def test_heaviside(self):
        x = np.array([[-30.0, -0.1, 0.0, 0.2], [7.5, np.nan, np.inf, -np.inf]])
        expectedhalf = np.array([[0.0, 0.0, 0.5, 1.0], [1.0, np.nan, 1.0, 0.0]])
        expected1 = expectedhalf.copy()
        expected1[0, 2] = 1
        h = ncu.heaviside(x, 0.5)
        assert_equal(h, expectedhalf)
        h = ncu.heaviside(x, 1.0)
        assert_equal(h, expected1)
        x = x.astype(np.float32)
        h = ncu.heaviside(x, np.float32(0.5))
        assert_equal(h, expectedhalf.astype(np.float32))
        h = ncu.heaviside(x, np.float32(1.0))
        assert_equal(h, expected1.astype(np.float32))