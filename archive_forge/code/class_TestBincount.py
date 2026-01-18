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
class TestBincount:

    def test_simple(self):
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    def test_simple2(self):
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    def test_simple_weight(self):
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    def test_simple_weight2(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    def test_with_minlength(self):
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))
        x = []
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([]))

    def test_with_minlength_smaller_than_maxvalue(self):
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    def test_with_minlength_and_weights(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

    def test_empty(self):
        x = np.array([], dtype=int)
        y = np.bincount(x)
        assert_array_equal(x, y)

    def test_empty_with_minlength(self):
        x = np.array([], dtype=int)
        y = np.bincount(x, minlength=5)
        assert_array_equal(y, np.zeros(5, dtype=int))

    def test_with_incorrect_minlength(self):
        x = np.array([], dtype=int)
        assert_raises_regex(TypeError, "'str' object cannot be interpreted", lambda: np.bincount(x, minlength='foobar'))
        assert_raises_regex(ValueError, 'must not be negative', lambda: np.bincount(x, minlength=-1))
        x = np.arange(5)
        assert_raises_regex(TypeError, "'str' object cannot be interpreted", lambda: np.bincount(x, minlength='foobar'))
        assert_raises_regex(ValueError, 'must not be negative', lambda: np.bincount(x, minlength=-1))

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    def test_dtype_reference_leaks(self):
        intp_refcount = sys.getrefcount(np.dtype(np.intp))
        double_refcount = sys.getrefcount(np.dtype(np.double))
        for j in range(10):
            np.bincount([1, 2, 3])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)
        for j in range(10):
            np.bincount([1, 2, 3], [4, 5, 6])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

    @pytest.mark.parametrize('vals', [[[2, 2]], 2])
    def test_error_not_1d(self, vals):
        vals_arr = np.asarray(vals)
        with assert_raises(ValueError):
            np.bincount(vals_arr)
        with assert_raises(ValueError):
            np.bincount(vals)