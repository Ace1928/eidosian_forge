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
class TestAngle:

    def test_basic(self):
        x = [1 + 3j, np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2, 1, 1j, -1, -1j, 1 - 3j, -1 + 3j]
        y = angle(x)
        yo = [np.arctan(3.0 / 1.0), np.arctan(1.0), 0, np.pi / 2, np.pi, -np.pi / 2.0, -np.arctan(3.0 / 1.0), np.pi - np.arctan(3.0 / 1.0)]
        z = angle(x, deg=True)
        zo = np.array(yo) * 180 / np.pi
        assert_array_almost_equal(y, yo, 11)
        assert_array_almost_equal(z, zo, 11)

    def test_subclass(self):
        x = np.ma.array([1 + 3j, 1, np.sqrt(2) / 2 * (1 + 1j)])
        x[1] = np.ma.masked
        expected = np.ma.array([np.arctan(3.0 / 1.0), 0, np.arctan(1.0)])
        expected[1] = np.ma.masked
        actual = angle(x)
        assert_equal(type(actual), type(expected))
        assert_equal(actual.mask, expected.mask)
        assert_equal(actual, expected)