import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestCApiAccess:

    def test_getitem(self):
        subscript = functools.partial(array_indexing, 0)
        assert_raises(IndexError, subscript, np.ones(()), 0)
        assert_raises(IndexError, subscript, np.ones(10), 11)
        assert_raises(IndexError, subscript, np.ones(10), -11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), 11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), -11)
        a = np.arange(10)
        assert_array_equal(a[4], subscript(a, 4))
        a = a.reshape(5, 2)
        assert_array_equal(a[-4], subscript(a, -4))

    def test_setitem(self):
        assign = functools.partial(array_indexing, 1)
        assert_raises(ValueError, assign, np.ones(10), 0)
        assert_raises(IndexError, assign, np.ones(()), 0, 0)
        assert_raises(IndexError, assign, np.ones(10), 11, 0)
        assert_raises(IndexError, assign, np.ones(10), -11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), 11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), -11, 0)
        a = np.arange(10)
        assign(a, 4, 10)
        assert_(a[4] == 10)
        a = a.reshape(5, 2)
        assign(a, 4, 10)
        assert_array_equal(a[-1], [10, 10])