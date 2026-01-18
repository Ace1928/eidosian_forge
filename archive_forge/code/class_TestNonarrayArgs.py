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
class TestNonarrayArgs:

    def test_choose(self):
        choices = [[0, 1, 2], [3, 4, 5], [5, 6, 7]]
        tgt = [5, 1, 5]
        a = [2, 0, 1]
        out = np.choose(a, choices)
        assert_equal(out, tgt)

    def test_clip(self):
        arr = [-1, 5, 2, 3, 10, -4, -9]
        out = np.clip(arr, 2, 7)
        tgt = [2, 5, 2, 3, 7, 2, 2]
        assert_equal(out, tgt)

    def test_compress(self):
        arr = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

    def test_count_nonzero(self):
        arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        assert_equal(out, tgt)

    def test_cumproduct(self):
        A = [[1, 2, 3], [4, 5, 6]]
        with assert_warns(DeprecationWarning):
            expected = np.array([1, 2, 6, 24, 120, 720])
            assert_(np.all(np.cumproduct(A) == expected))

    def test_diagonal(self):
        a = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        out = np.diagonal(a)
        tgt = [0, 5, 10]
        assert_equal(out, tgt)

    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.mean(A) == 3.5)
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        assert_(np.all(np.mean(A, 1) == np.array([2.0, 5.0])))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.mean([])))
            assert_(w[0].category is RuntimeWarning)

    def test_ptp(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.ptp(a, axis=0), 15.0)

    def test_prod(self):
        arr = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        tgt = [24, 1890, 600]
        assert_equal(np.prod(arr, axis=-1), tgt)

    def test_ravel(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    def test_repeat(self):
        a = [1, 2, 3]
        tgt = [1, 1, 2, 2, 3, 3]
        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    def test_reshape(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    def test_round(self):
        arr = [1.56, 72.54, 6.35, 3.25]
        tgt = [1.6, 72.5, 6.4, 3.2]
        assert_equal(np.around(arr, decimals=1), tgt)
        s = np.float64(1.0)
        assert_(isinstance(s.round(), np.float64))
        assert_equal(s.round(), 1.0)

    @pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64])
    def test_dunder_round(self, dtype):
        s = dtype(1)
        assert_(isinstance(round(s), int))
        assert_(isinstance(round(s, None), int))
        assert_(isinstance(round(s, ndigits=None), int))
        assert_equal(round(s), 1)
        assert_equal(round(s, None), 1)
        assert_equal(round(s, ndigits=None), 1)

    @pytest.mark.parametrize('val, ndigits', [pytest.param(2 ** 31 - 1, -1, marks=pytest.mark.xfail(reason='Out of range of int32')), (2 ** 31 - 1, 1 - math.ceil(math.log10(2 ** 31 - 1))), (2 ** 31 - 1, -math.ceil(math.log10(2 ** 31 - 1)))])
    def test_dunder_round_edgecases(self, val, ndigits):
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    def test_dunder_round_accuracy(self):
        f = np.float64(5.1 * 10 ** 73)
        assert_(isinstance(round(f, -73), np.float64))
        assert_array_max_ulp(round(f, -73), 5.0 * 10 ** 73)
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10 ** 73)
        i = np.int64(501)
        assert_(isinstance(round(i, -2), np.int64))
        assert_array_max_ulp(round(i, -2), 500)
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @pytest.mark.xfail(raises=AssertionError, reason='gh-15896')
    def test_round_py_consistency(self):
        f = 5.1 * 10 ** 73
        assert_equal(round(np.float64(f), -73), round(f, -73))

    def test_searchsorted(self):
        arr = [-8, -5, -1, 3, 6, 10]
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    def test_size(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.size(A) == 6)
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)

    def test_squeeze(self):
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_std(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.std(A), 1.707825127659933)
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.std([])))
            assert_(w[0].category is RuntimeWarning)

    def test_swapaxes(self):
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)
        assert_equal(tgt, out)

    def test_take(self):
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]
        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        c = [[1, 2], [3, 4], [5, 6]]
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.var(A), 2.9166666666666665)
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.var([])))
            assert_(w[0].category is RuntimeWarning)
        B = np.array([None, 0])
        B[0] = 1j
        assert_almost_equal(np.var(B), 0.25)