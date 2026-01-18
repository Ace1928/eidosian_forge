import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestFloatNonIntegerArgument:
    """
    These test that ``TypeError`` is raised when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    """

    def test_valid_indexing(self):
        a = np.array([[[5]]])
        a[np.array([0])]
        a[[0, 0]]
        a[:, [0, 0]]
        a[:, 0, :]
        a[:, :, :]

    def test_valid_slicing(self):
        a = np.array([[[5]]])
        a[:]
        a[0:]
        a[:2]
        a[0:2]
        a[::2]
        a[1::2]
        a[:2:2]
        a[1:2:2]

    def test_non_integer_argument_errors(self):
        a = np.array([[5]])
        assert_raises(TypeError, np.reshape, a, (1.0, 1.0, -1))
        assert_raises(TypeError, np.reshape, a, (np.array(1.0), -1))
        assert_raises(TypeError, np.take, a, [0], 1.0)
        assert_raises(TypeError, np.take, a, [0], np.float64(1.0))

    def test_non_integer_sequence_multiplication(self):

        def mult(a, b):
            return a * b
        assert_raises(TypeError, mult, [1], np.float_(3))
        mult([1], np.int_(3))

    def test_reduce_axis_float_index(self):
        d = np.zeros((3, 3, 3))
        assert_raises(TypeError, np.min, d, 0.5)
        assert_raises(TypeError, np.min, d, (0.5, 1))
        assert_raises(TypeError, np.min, d, (1, 2.2))
        assert_raises(TypeError, np.min, d, (0.2, 1.2))