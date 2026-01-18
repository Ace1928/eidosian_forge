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
class TestAverage:

    def test_basic(self):
        y1 = np.array([1, 2, 3])
        assert_(average(y1, axis=0) == 2.0)
        y2 = np.array([1.0, 2.0, 3.0])
        assert_(average(y2, axis=0) == 2.0)
        y3 = [0.0, 0.0, 0.0]
        assert_(average(y3, axis=0) == 0.0)
        y4 = np.ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        assert_almost_equal(y4.mean(0), average(y4, 0))
        assert_almost_equal(y4.mean(1), average(y4, 1))
        y5 = rand(5, 5)
        assert_almost_equal(y5.mean(0), average(y5, 0))
        assert_almost_equal(y5.mean(1), average(y5, 1))

    @pytest.mark.parametrize('x, axis, expected_avg, weights, expected_wavg, expected_wsum', [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]), ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]], [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])])
    def test_basic_keepdims(self, x, axis, expected_avg, weights, expected_wavg, expected_wsum):
        avg = np.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)
        wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        wavg, wsum = np.average(x, axis=axis, weights=weights, returned=True, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    def test_weights(self):
        y = np.arange(10)
        w = np.arange(10)
        actual = average(y, weights=w)
        desired = (np.arange(10) ** 2).sum() * 1.0 / np.arange(10).sum()
        assert_almost_equal(actual, desired)
        y1 = np.array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        actual = average(y1, weights=w0, axis=0)
        desired = np.array([3.0, 4.0, 5.0])
        assert_almost_equal(actual, desired)
        w1 = [0, 0, 1]
        actual = average(y1, weights=w1, axis=1)
        desired = np.array([3.0, 6.0])
        assert_almost_equal(actual, desired)
        w2 = [[0, 0, 1], [0, 0, 2]]
        desired = np.array([3.0, 6.0])
        assert_array_equal(average(y1, weights=w2, axis=1), desired)
        assert_equal(average(y1, weights=w2), 5.0)
        y3 = rand(5).astype(np.float32)
        w3 = rand(5).astype(np.float64)
        assert_(np.average(y3, weights=w3).dtype == np.result_type(y3, w3))
        x = np.array([2, 3, 4]).reshape(3, 1)
        w = np.array([4, 5, 6]).reshape(3, 1)
        actual = np.average(x, weights=w, axis=1, keepdims=False)
        desired = np.array([2.0, 3.0, 4.0])
        assert_array_equal(actual, desired)
        actual = np.average(x, weights=w, axis=1, keepdims=True)
        desired = np.array([[2.0], [3.0], [4.0]])
        assert_array_equal(actual, desired)

    def test_returned(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])
        avg, scl = average(y, returned=True)
        assert_equal(scl, 6.0)
        avg, scl = average(y, 0, returned=True)
        assert_array_equal(scl, np.array([2.0, 2.0, 2.0]))
        avg, scl = average(y, 1, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0]))
        w0 = [1, 2]
        avg, scl = average(y, weights=w0, axis=0, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0, 3.0]))
        w1 = [1, 2, 3]
        avg, scl = average(y, weights=w1, axis=1, returned=True)
        assert_array_equal(scl, np.array([6.0, 6.0]))
        w2 = [[0, 0, 1], [1, 2, 3]]
        avg, scl = average(y, weights=w2, axis=1, returned=True)
        assert_array_equal(scl, np.array([1.0, 6.0]))

    def test_subclasses(self):

        class subclass(np.ndarray):
            pass
        a = np.array([[1, 2], [3, 4]]).view(subclass)
        w = np.array([[1, 2], [3, 4]]).view(subclass)
        assert_equal(type(np.average(a)), subclass)
        assert_equal(type(np.average(a, weights=w)), subclass)

    def test_upcasting(self):
        typs = [('i4', 'i4', 'f8'), ('i4', 'f4', 'f8'), ('f4', 'i4', 'f8'), ('f4', 'f4', 'f4'), ('f4', 'f8', 'f8')]
        for at, wt, rt in typs:
            a = np.array([[1, 2], [3, 4]], dtype=at)
            w = np.array([[1, 2], [3, 4]], dtype=wt)
            assert_equal(np.average(a, weights=w).dtype, np.dtype(rt))

    def test_object_dtype(self):
        a = np.array([decimal.Decimal(x) for x in range(10)])
        w = np.array([decimal.Decimal(1) for _ in range(10)])
        w /= w.sum()
        assert_almost_equal(a.mean(0), average(a, weights=w))

    def test_average_class_without_dtype(self):
        a = np.array([Fraction(1, 5), Fraction(3, 5)])
        assert_equal(np.average(a), Fraction(2, 5))