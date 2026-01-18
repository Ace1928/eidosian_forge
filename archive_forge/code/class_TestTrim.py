import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestTrim:

    def test_trim1(self):
        a = np.arange(11)
        assert_equal(np.sort(stats.trim1(a, 0.1)), np.arange(10))
        assert_equal(np.sort(stats.trim1(a, 0.2)), np.arange(9))
        assert_equal(np.sort(stats.trim1(a, 0.2, tail='left')), np.arange(2, 11))
        assert_equal(np.sort(stats.trim1(a, 3 / 11.0, tail='left')), np.arange(3, 11))
        assert_equal(stats.trim1(a, 1.0), [])
        assert_equal(stats.trim1(a, 1.0, tail='left'), [])
        assert_equal(stats.trim1([], 0.1), [])
        assert_equal(stats.trim1([], 3 / 11.0, tail='left'), [])
        assert_equal(stats.trim1([], 4 / 6.0), [])
        a = np.arange(24).reshape(6, 4)
        ref = np.arange(4, 24).reshape(5, 4)
        axis = 0
        trimmed = stats.trim1(a, 0.2, tail='left', axis=axis)
        assert_equal(np.sort(trimmed, axis=axis), ref)
        axis = 1
        trimmed = stats.trim1(a.T, 0.2, tail='left', axis=axis)
        assert_equal(np.sort(trimmed, axis=axis), ref.T)

    def test_trimboth(self):
        a = np.arange(11)
        assert_equal(np.sort(stats.trimboth(a, 3 / 11.0)), np.arange(3, 8))
        assert_equal(np.sort(stats.trimboth(a, 0.2)), np.array([2, 3, 4, 5, 6, 7, 8]))
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(6, 4), 0.2)), np.arange(4, 20).reshape(4, 4))
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(4, 6).T, 2 / 6.0)), np.array([[2, 8, 14, 20], [3, 9, 15, 21]]))
        assert_raises(ValueError, stats.trimboth, np.arange(24).reshape(4, 6).T, 4 / 6.0)
        assert_equal(stats.trimboth([], 0.1), [])
        assert_equal(stats.trimboth([], 3 / 11.0), [])
        assert_equal(stats.trimboth([], 4 / 6.0), [])

    def test_trim_mean(self):
        a = np.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
        idx = np.array([3, 5, 0, 1, 2, 4])
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        assert_equal(stats.trim_mean(a3, 2 / 6.0), np.array([2.5, 8.5, 14.5, 20.5]))
        assert_equal(stats.trim_mean(a2, 2 / 6.0), np.array([10.0, 11.0, 12.0, 13.0]))
        idx4 = np.array([1, 0, 3, 2])
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        assert_equal(stats.trim_mean(a4, 2 / 6.0), np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0]))
        a = [7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23, 20, 2, 14, 4, 13, 8, 3]
        assert_equal(stats.trim_mean(a, 2 / 6.0), 11.5)
        assert_equal(stats.trim_mean([5, 4, 3, 1, 2, 0], 2 / 6.0), 2.5)
        np.random.seed(1234)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        for axis in [0, 1, 2, 3, -1]:
            res1 = stats.trim_mean(a, 2 / 6.0, axis=axis)
            res2 = stats.trim_mean(np.moveaxis(a, axis, 0), 2 / 6.0)
            assert_equal(res1, res2)
        res1 = stats.trim_mean(a, 2 / 6.0, axis=None)
        res2 = stats.trim_mean(a.ravel(), 2 / 6.0)
        assert_equal(res1, res2)
        assert_raises(ValueError, stats.trim_mean, a, 0.6)
        assert_equal(stats.trim_mean([], 0.0), np.nan)
        assert_equal(stats.trim_mean([], 0.6), np.nan)