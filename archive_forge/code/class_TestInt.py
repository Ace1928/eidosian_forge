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
class TestInt:

    def test_logical_not(self):
        x = np.ones(10, dtype=np.int16)
        o = np.ones(10 * 2, dtype=bool)
        tgt = o.copy()
        tgt[::2] = False
        os = o[::2]
        assert_array_equal(np.logical_not(x, out=os), False)
        assert_array_equal(o, tgt)