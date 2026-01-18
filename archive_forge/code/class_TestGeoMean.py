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
class TestGeoMean:

    def test_0(self):
        a = [1, 0, 2]
        desired = 0
        check_equal_gmean(a, desired)

    def test_1d_list(self):
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        desired = 45.2872868812
        check_equal_gmean(a, desired)
        a = [1, 2, 3, 4]
        desired = power(1 * 2 * 3 * 4, 1.0 / 4.0)
        check_equal_gmean(a, desired, rtol=1e-14)

    def test_1d_array(self):
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        desired = 45.2872868812
        check_equal_gmean(a, desired)
        a = array([1, 2, 3, 4], float32)
        desired = power(1 * 2 * 3 * 4, 1.0 / 4.0)
        check_equal_gmean(a, desired, dtype=float32)

    def test_2d_list(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        check_equal_gmean(a, desired)

    def test_2d_array(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 52.8885199
        check_equal_gmean(array(a), desired)

    def test_2d_axis0(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([35.56893304, 49.32424149, 61.3579244, 72.68482371])
        check_equal_gmean(a, desired, axis=0)
        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        desired = array([1, 2, 3, 4])
        check_equal_gmean(a, desired, axis=0, rtol=1e-14)

    def test_2d_axis1(self):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.13363839, 64.02171746, 104.40086817])
        check_equal_gmean(a, desired, axis=1)
        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        v = power(1 * 2 * 3 * 4, 1.0 / 4.0)
        desired = array([v, v, v])
        check_equal_gmean(a, desired, axis=1, rtol=1e-14)

    def test_large_values(self):
        a = array([1e+100, 1e+200, 1e+300])
        desired = 1e+200
        check_equal_gmean(a, desired, rtol=1e-13)

    def test_1d_list0(self):
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 0]
        desired = 0.0
        with np.errstate(all='ignore'):
            check_equal_gmean(a, desired)

    def test_1d_array0(self):
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 0])
        desired = 0.0
        with np.errstate(divide='ignore'):
            check_equal_gmean(a, desired)

    def test_1d_list_neg(self):
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
        desired = np.nan
        with np.errstate(invalid='ignore'):
            check_equal_gmean(a, desired)

    def test_weights_1d_list(self):
        a = [1, 2, 3, 4, 5]
        weights = [2, 5, 6, 4, 3]
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-05)

    def test_weights_1d_array(self):
        a = np.array([1, 2, 3, 4, 5])
        weights = np.array([2, 5, 6, 4, 3])
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-05)

    def test_weights_masked_1d_array(self):
        a = np.array([1, 2, 3, 4, 5, 6])
        weights = np.ma.array([2, 5, 6, 4, 3, 5], mask=[0, 0, 0, 0, 0, 1])
        desired = 2.77748
        check_equal_gmean(a, desired, weights=weights, rtol=1e-05)