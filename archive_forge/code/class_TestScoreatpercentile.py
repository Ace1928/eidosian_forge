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
class TestScoreatpercentile:

    def setup_method(self):
        self.a1 = [3, 4, 5, 10, -3, -5, 6]
        self.a2 = [3, -6, -2, 8, 7, 4, 2, 1]
        self.a3 = [3.0, 4, 5, 10, -3, -5, -6, 7.0]

    def test_basic(self):
        x = arange(8) * 0.5
        assert_equal(stats.scoreatpercentile(x, 0), 0.0)
        assert_equal(stats.scoreatpercentile(x, 100), 3.5)
        assert_equal(stats.scoreatpercentile(x, 50), 1.75)

    def test_fraction(self):
        scoreatperc = stats.scoreatpercentile
        assert_equal(scoreatperc(list(range(10)), 50), 4.5)
        assert_equal(scoreatperc(list(range(10)), 50, (2, 7)), 4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8)), 4.5)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (10, 100)), 55)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (1, 10)), 5.5)
        assert_equal(scoreatperc(list(range(10)), 50, interpolation_method='fraction'), 4.5)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2, 7), interpolation_method='fraction'), 4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8), interpolation_method='fraction'), 4.5)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (10, 100), interpolation_method='fraction'), 55)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (1, 10), interpolation_method='fraction'), 5.5)

    def test_lower_higher(self):
        scoreatperc = stats.scoreatpercentile
        assert_equal(scoreatperc(list(range(10)), 50, interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50, interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(10)), 50, (2, 7), interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2, 7), interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(100)), 50, (1, 8), interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(100)), 50, (1, 8), interpolation_method='higher'), 5)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (10, 100), interpolation_method='lower'), 10)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(10, 100), interpolation_method='higher'), 100)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (1, 10), interpolation_method='lower'), 1)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(1, 10), interpolation_method='higher'), 10)

    def test_sequence_per(self):
        x = arange(8) * 0.5
        expected = np.array([0, 3.5, 1.75])
        res = stats.scoreatpercentile(x, [0, 100, 50])
        assert_allclose(res, expected)
        assert_(isinstance(res, np.ndarray))
        assert_allclose(stats.scoreatpercentile(x, np.array([0, 100, 50])), expected)
        res2 = stats.scoreatpercentile(np.arange(12).reshape((3, 4)), np.array([0, 1, 100, 100]), axis=1)
        expected2 = array([[0, 4, 8], [0.03, 4.03, 8.03], [3, 7, 11], [3, 7, 11]])
        assert_allclose(res2, expected2)

    def test_axis(self):
        scoreatperc = stats.scoreatpercentile
        x = arange(12).reshape(3, 4)
        assert_equal(scoreatperc(x, (25, 50, 100)), [2.75, 5.5, 11.0])
        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        assert_equal(scoreatperc(x, (25, 50, 100), axis=0), r0)
        r1 = [[0.75, 4.75, 8.75], [1.5, 5.5, 9.5], [3, 7, 11]]
        assert_equal(scoreatperc(x, (25, 50, 100), axis=1), r1)
        x = array([[1, 1, 1], [1, 1, 1], [4, 4, 3], [1, 1, 1], [1, 1, 1]])
        score = stats.scoreatpercentile(x, 50)
        assert_equal(score.shape, ())
        assert_equal(score, 1.0)
        score = stats.scoreatpercentile(x, 50, axis=0)
        assert_equal(score.shape, (3,))
        assert_equal(score, [1, 1, 1])

    def test_exception(self):
        assert_raises(ValueError, stats.scoreatpercentile, [1, 2], 56, interpolation_method='foobar')
        assert_raises(ValueError, stats.scoreatpercentile, [1], 101)
        assert_raises(ValueError, stats.scoreatpercentile, [1], -1)

    def test_empty(self):
        assert_equal(stats.scoreatpercentile([], 50), np.nan)
        assert_equal(stats.scoreatpercentile(np.array([[], []]), 50), np.nan)
        assert_equal(stats.scoreatpercentile([], [50, 99]), [np.nan, np.nan])