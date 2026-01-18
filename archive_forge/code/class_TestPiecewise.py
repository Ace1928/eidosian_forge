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
class TestPiecewise:

    def test_simple(self):
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])
        x = piecewise([0, 0], np.array([True, False]), [1])
        assert_array_equal(x, [1, 0])
        x = piecewise([0, 0], np.array([1, 0]), [1])
        assert_array_equal(x, [1, 0])
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])
        x = piecewise([0, 0], [[False, True]], [lambda x: -1])
        assert_array_equal(x, [0, -1])
        assert_raises_regex(ValueError, '1 or 2 functions are expected', piecewise, [0, 0], [[False, True]], [])
        assert_raises_regex(ValueError, '1 or 2 functions are expected', piecewise, [0, 0], [[False, True]], [1, 2, 3])

    def test_two_conditions(self):
        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_scalar_domains_three_conditions(self):
        x = piecewise(3, [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    def test_default(self):
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)
        x = 5
        y = piecewise(x, [True, False], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)
        y = piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    def test_0d_comparison(self):
        x = 3
        y = piecewise(x, [x <= 3, x > 3], [4, 0])
        assert_equal(y, 4)
        x = 4
        y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)
        assert_raises_regex(ValueError, '2 or 3 functions are expected', piecewise, x, [x <= 3, x > 3], [1])
        assert_raises_regex(ValueError, '2 or 3 functions are expected', piecewise, x, [x <= 3, x > 3], [1, 1, 1, 1])

    def test_0d_0d_condition(self):
        x = np.array(3)
        c = np.array(x > 3)
        y = piecewise(x, [c], [1, 2])
        assert_equal(y, 2)

    def test_multidimensional_extrafunc(self):
        x = np.array([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        y = piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        assert_array_equal(y, np.array([[-1.0, -1.0, -1.0], [3.0, 3.0, 1.0]]))

    def test_subclasses(self):

        class subclass(np.ndarray):
            pass
        x = np.arange(5.0).view(subclass)
        r = piecewise(x, [x < 2.0, x >= 4], [-1.0, 1.0, 0.0])
        assert_equal(type(r), subclass)
        assert_equal(r, [-1.0, -1.0, 0.0, 0.0, 1.0])