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
class TestFlip:

    def test_axes(self):
        assert_raises(np.AxisError, np.flip, np.ones(4), axis=1)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=2)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=-3)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))

    def test_basic_lr(self):
        a = get_mat(4)
        b = a[:, ::-1]
        assert_equal(np.flip(a, 1), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[2, 1, 0], [5, 4, 3]]
        assert_equal(np.flip(a, 1), b)

    def test_basic_ud(self):
        a = get_mat(4)
        b = a[::-1, :]
        assert_equal(np.flip(a, 0), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[3, 4, 5], [0, 1, 2]]
        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis0(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        b = np.array([[[4, 5], [6, 7]], [[0, 1], [2, 3]]])
        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis1(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        b = np.array([[[2, 3], [0, 1]], [[6, 7], [4, 5]]])
        assert_equal(np.flip(a, 1), b)

    def test_3d_swap_axis2(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        b = np.array([[[1, 0], [3, 2]], [[5, 4], [7, 6]]])
        assert_equal(np.flip(a, 2), b)

    def test_4d(self):
        a = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        for i in range(a.ndim):
            assert_equal(np.flip(a, i), np.flipud(a.swapaxes(0, i)).swapaxes(i, 0))

    def test_default_axis(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[6, 5, 4], [3, 2, 1]])
        assert_equal(np.flip(a), b)

    def test_multiple_axes(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        assert_equal(np.flip(a, axis=()), a)
        b = np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])
        assert_equal(np.flip(a, axis=(0, 2)), b)
        c = np.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]])
        assert_equal(np.flip(a, axis=(1, 2)), c)