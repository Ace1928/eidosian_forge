import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class _GenericTest:

    def _test_equal(self, a, b):
        self._assert_func(a, b)

    def _test_not_equal(self, a, b):
        with assert_raises(AssertionError):
            self._assert_func(a, b)

    def test_array_rank1_eq(self):
        """Test two equal array of rank 1 are found equal."""
        a = np.array([1, 2])
        b = np.array([1, 2])
        self._test_equal(a, b)

    def test_array_rank1_noteq(self):
        """Test two different array of rank 1 are found not equal."""
        a = np.array([1, 2])
        b = np.array([2, 2])
        self._test_not_equal(a, b)

    def test_array_rank2_eq(self):
        """Test two equal array of rank 2 are found equal."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])
        self._test_equal(a, b)

    def test_array_diffshape(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array([1, 2])
        b = np.array([[1, 2], [1, 2]])
        self._test_not_equal(a, b)

    def test_objarray(self):
        """Test object arrays."""
        a = np.array([1, 1], dtype=object)
        self._test_equal(a, 1)

    def test_array_likes(self):
        self._test_equal([1, 2, 3], (1, 2, 3))