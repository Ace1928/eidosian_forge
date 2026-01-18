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
class TestHarMean:

    def test_0(self):
        a = [1, 0, 2]
        desired = 0
        check_equal_hmean(a, desired)

    def test_1d_list(self):
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        desired = 34.1417152147
        check_equal_hmean(a, desired)
        a = [1, 2, 3, 4]
        desired = 4.0 / (1.0 / 1 + 1.0 / 2 + 1.0 / 3 + 1.0 / 4)
        check_equal_hmean(a, desired)

    def test_1d_array(self):
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 34.1417152147
        check_equal_hmean(a, desired)

    def test_1d_array_with_zero(self):
        a = np.array([1, 0])
        desired = 0.0
        assert_equal(stats.hmean(a), desired)

    def test_1d_array_with_negative_value(self):
        a = np.array([1, 0, -1])
        assert_raises(ValueError, stats.hmean, a)

    def test_2d_list(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(a, desired)

    def test_2d_array(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(np.array(a), desired)

    def test_2d_axis0(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 39.13043478, 52.90076336, 65.45454545])
        check_equal_hmean(a, desired, axis=0)

    def test_2d_axis0_with_zero(self):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 0.0, 52.90076336, 65.45454545])
        assert_allclose(stats.hmean(a, axis=0), desired)

    def test_2d_axis1(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([19.2, 63.03939962, 103.80078637])
        check_equal_hmean(a, desired, axis=1)

    def test_2d_axis1_with_zero(self):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([0.0, 63.03939962, 103.80078637])
        assert_allclose(stats.hmean(a, axis=1), desired)

    def test_weights_1d_list(self):
        a = [2, 10, 6]
        weights = [10, 5, 3]
        desired = 3
        check_equal_hmean(a, desired, weights=weights, rtol=1e-05)

    def test_weights_2d_array_axis0(self):
        a = np.array([[2, 5], [10, 5], [6, 5]])
        weights = np.array([[10, 1], [5, 1], [3, 1]])
        desired = np.array([3, 5])
        check_equal_hmean(a, desired, axis=0, weights=weights, rtol=1e-05)

    def test_weights_2d_array_axis1(self):
        a = np.array([[2, 10, 6], [7, 7, 7]])
        weights = np.array([[10, 5, 3], [1, 1, 1]])
        desired = np.array([3, 7])
        check_equal_hmean(a, desired, axis=1, weights=weights, rtol=1e-05)

    def test_weights_masked_1d_array(self):
        a = np.array([2, 10, 6, 42])
        weights = np.ma.array([10, 5, 3, 42], mask=[0, 0, 0, 1])
        desired = 3
        check_equal_hmean(a, desired, weights=weights, rtol=1e-05)