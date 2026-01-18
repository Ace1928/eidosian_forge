import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestCumprod:

    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64, np.complex64, np.complex128]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, np.cumprod, a)
                assert_raises(ArithmeticError, np.cumprod, a2, 1)
                assert_raises(ArithmeticError, np.cumprod, a)
            else:
                assert_array_equal(np.cumprod(a, axis=-1), np.array([1, 2, 20, 220, 1320, 6600, 26400], ctype))
                assert_array_equal(np.cumprod(a2, axis=0), np.array([[1, 2, 3, 4], [5, 12, 21, 36], [50, 36, 84, 180]], ctype))
                assert_array_equal(np.cumprod(a2, axis=-1), np.array([[1, 2, 6, 24], [5, 30, 210, 1890], [10, 30, 120, 600]], ctype))