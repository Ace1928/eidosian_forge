import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestFancyIndexingEquivalence:

    def test_object_assign(self):
        a = np.arange(5, dtype=object)
        b = a.copy()
        a[:3] = [1, (1, 2), 3]
        b[[0, 1, 2]] = [1, (1, 2), 3]
        assert_array_equal(a, b)
        b = np.arange(5, dtype=object)[None, :]
        b[[0], :3] = [[1, (1, 2), 3]]
        assert_array_equal(a, b[0])
        b = b.T
        b[:3, [0]] = [[1], [(1, 2)], [3]]
        assert_array_equal(a, b[:, 0])
        arr = np.ones((3, 4, 5), dtype=object)
        cmp_arr = arr.copy()
        cmp_arr[:1, ...] = [[[1], [2], [3], [4]]]
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)
        arr = arr.copy('F')
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)

    def test_cast_equivalence(self):
        a = np.arange(5)
        b = a.copy()
        a[:3] = np.array(['2', '-3', '-1'])
        b[[0, 2, 1]] = np.array(['2', '-1', '-3'])
        assert_array_equal(a, b)
        b = np.arange(5)[None, :]
        b[[0], :3] = np.array([['2', '-3', '-1']])
        assert_array_equal(a, b[0])