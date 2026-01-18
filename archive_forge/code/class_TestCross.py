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
class TestCross:

    def test_2x2(self):
        u = [1, 2]
        v = [3, 4]
        z = -2
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_2x3(self):
        u = [1, 2]
        v = [3, 4, 5]
        z = np.array([10, -5, -2])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_3x3(self):
        u = [1, 2, 3]
        v = [4, 5, 6]
        z = np.array([-3, 6, -3])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_broadcasting(self):
        u = np.tile([1, 2], (11, 1))
        v = np.tile([3, 4], (11, 1))
        z = -2
        assert_equal(np.cross(u, v), z)
        assert_equal(np.cross(v, u), -z)
        assert_equal(np.cross(u, u), 0)
        u = np.tile([1, 2], (11, 1)).T
        v = np.tile([3, 4, 5], (11, 1))
        z = np.tile([10, -5, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0), z)
        assert_equal(np.cross(v, u.T), -z)
        assert_equal(np.cross(v, v), 0)
        u = np.tile([1, 2, 3], (11, 1)).T
        v = np.tile([3, 4], (11, 1)).T
        z = np.tile([-12, 9, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)
        assert_equal(np.cross(v.T, u.T), -z)
        assert_equal(np.cross(u.T, u.T), 0)
        u = np.tile([1, 2, 3], (5, 1))
        v = np.tile([4, 5, 6], (5, 1)).T
        z = np.tile([-3, 6, -3], (5, 1))
        assert_equal(np.cross(u, v, axisb=0), z)
        assert_equal(np.cross(v.T, u), -z)
        assert_equal(np.cross(u, u), 0)

    def test_broadcasting_shapes(self):
        u = np.ones((2, 1, 3))
        v = np.ones((5, 3))
        assert_equal(np.cross(u, v).shape, (2, 5, 3))
        u = np.ones((10, 3, 5))
        v = np.ones((2, 5))
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
        u = np.ones((10, 3, 5, 7))
        v = np.ones((5, 7, 2))
        assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
        assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
        u = np.ones((3, 4, 2))
        for axisc in range(-2, 2):
            assert_equal(np.cross(u, u, axisc=axisc).shape, (3, 4))

    def test_uint8_int32_mixed_dtypes(self):
        u = np.array([[195, 8, 9]], np.uint8)
        v = np.array([250, 166, 68], np.int32)
        z = np.array([[950, 11010, -30370]], dtype=np.int32)
        assert_equal(np.cross(v, u), z)
        assert_equal(np.cross(u, v), -z)