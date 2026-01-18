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
class TestWassersteinDistance:
    """ Tests for wasserstein_distance() output values.
    """

    def test_simple(self):
        assert_almost_equal(stats.wasserstein_distance([0, 1], [0], [1, 1], [1]), 0.5)
        assert_almost_equal(stats.wasserstein_distance([0, 1], [0], [3, 1], [1]), 0.25)
        assert_almost_equal(stats.wasserstein_distance([0, 2], [0], [1, 1], [1]), 1)
        assert_almost_equal(stats.wasserstein_distance([0, 1, 2], [1, 2, 3]), 1)

    def test_same_distribution(self):
        assert_equal(stats.wasserstein_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(stats.wasserstein_distance([1, 1, 1, 4], [4, 1], [1, 1, 1, 1], [1, 3]), 0)

    def test_shift(self):
        assert_almost_equal(stats.wasserstein_distance([0], [1]), 1)
        assert_almost_equal(stats.wasserstein_distance([-5], [5]), 10)
        assert_almost_equal(stats.wasserstein_distance([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]), 10)
        assert_almost_equal(stats.wasserstein_distance([4.5, 6.7, 2.1], [4.6, 7, 9.2], [3, 1, 1], [1, 3, 1]), 2.5)

    def test_combine_weights(self):
        assert_almost_equal(stats.wasserstein_distance([0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]), stats.wasserstein_distance([5, 0, 1], [0, 4, 3], [1, 2, 4], [1, 2, 4]))

    def test_collapse(self):
        u = np.arange(-10, 30, 0.3)
        v = np.zeros_like(u)
        assert_almost_equal(stats.wasserstein_distance(u, v), np.mean(np.abs(u)))
        u_weights = np.arange(len(u))
        v_weights = u_weights[::-1]
        assert_almost_equal(stats.wasserstein_distance(u, v, u_weights, v_weights), np.average(np.abs(u), weights=u_weights))

    def test_zero_weight(self):
        assert_almost_equal(stats.wasserstein_distance([1, 2, 100000], [1, 1], [1, 1, 0], [1, 1]), stats.wasserstein_distance([1, 2], [1, 1], [1, 1], [1, 1]))

    def test_inf_values(self):
        assert_equal(stats.wasserstein_distance([1, 2, np.inf], [1, 1]), np.inf)
        assert_equal(stats.wasserstein_distance([1, 2, np.inf], [-np.inf, 1]), np.inf)
        assert_equal(stats.wasserstein_distance([1, -np.inf, np.inf], [1, 1]), np.inf)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'invalid value*')
            assert_equal(stats.wasserstein_distance([1, 2, np.inf], [np.inf, 1]), np.nan)