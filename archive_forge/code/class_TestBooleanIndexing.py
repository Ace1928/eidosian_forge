import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestBooleanIndexing:

    def test_bool_as_int_argument_errors(self):
        a = np.array([[[1]]])
        assert_raises(TypeError, np.reshape, a, (True, -1))
        assert_raises(TypeError, np.reshape, a, (np.bool_(True), -1))
        assert_raises(TypeError, operator.index, np.array(True))
        assert_warns(DeprecationWarning, operator.index, np.True_)
        assert_raises(TypeError, np.take, args=(a, [0], False))

    def test_boolean_indexing_weirdness(self):
        a = np.ones((2, 3, 4))
        assert a[False, True, ...].shape == (0, 2, 3, 4)
        assert a[True, [0, 1], True, True, [1], [[2]]].shape == (1, 2)
        assert_raises(IndexError, lambda: a[False, [0, 1], ...])

    def test_boolean_indexing_fast_path(self):
        a = np.ones((3, 3))
        idx1 = np.array([[False] * 9])
        assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx1])
        idx2 = np.array([[False] * 8 + [True]])
        assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx2])
        idx3 = np.array([[False] * 10])
        assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1', lambda: a[idx3])
        a = np.ones((1, 1, 2))
        idx = np.array([[[True], [False]]])
        assert_raises_regex(IndexError, 'boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 2', lambda: a[idx])