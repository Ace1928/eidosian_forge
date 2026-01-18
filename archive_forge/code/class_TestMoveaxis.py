import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
class TestMoveaxis:

    def test_move_to_end(self):
        x = np.random.randn(5, 6, 7)
        for source, expected in [(0, (6, 7, 5)), (1, (5, 7, 6)), (2, (5, 6, 7)), (-1, (5, 6, 7))]:
            actual = np.moveaxis(x, source, -1).shape
            assert_(actual, expected)

    def test_move_new_position(self):
        x = np.random.randn(1, 2, 3, 4)
        for source, destination, expected in [(0, 1, (2, 1, 3, 4)), (1, 2, (1, 3, 2, 4)), (1, -1, (1, 3, 4, 2))]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_preserve_order(self):
        x = np.zeros((1, 2, 3, 4))
        for source, destination in [(0, 0), (3, -1), (-1, 3), ([0, -1], [0, -1]), ([2, 0], [2, 0]), (range(4), range(4))]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, (1, 2, 3, 4))

    def test_move_multiples(self):
        x = np.zeros((0, 1, 2, 3))
        for source, destination, expected in [([0, 1], [2, 3], (2, 3, 0, 1)), ([2, 3], [0, 1], (2, 3, 0, 1)), ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)), ([3, 0], [1, 0], (0, 3, 1, 2)), ([0, 3], [0, 1], (0, 3, 1, 2))]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_errors(self):
        x = np.random.randn(1, 2, 3)
        assert_raises_regex(np.AxisError, 'source.*out of bounds', np.moveaxis, x, 3, 0)
        assert_raises_regex(np.AxisError, 'source.*out of bounds', np.moveaxis, x, -4, 0)
        assert_raises_regex(np.AxisError, 'destination.*out of bounds', np.moveaxis, x, 0, 5)
        assert_raises_regex(ValueError, 'repeated axis in `source`', np.moveaxis, x, [0, 0], [0, 1])
        assert_raises_regex(ValueError, 'repeated axis in `destination`', np.moveaxis, x, [0, 1], [1, 1])
        assert_raises_regex(ValueError, 'must have the same number', np.moveaxis, x, 0, [0, 1])
        assert_raises_regex(ValueError, 'must have the same number', np.moveaxis, x, [0, 1], [0])

    def test_array_likes(self):
        x = np.ma.zeros((1, 2, 3))
        result = np.moveaxis(x, 0, 0)
        assert_(x.shape, result.shape)
        assert_(isinstance(result, np.ma.MaskedArray))
        x = [1, 2, 3]
        result = np.moveaxis(x, 0, 0)
        assert_(x, list(result))
        assert_(isinstance(result, np.ndarray))