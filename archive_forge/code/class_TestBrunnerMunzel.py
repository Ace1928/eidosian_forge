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
class TestBrunnerMunzel:
    X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13

    def test_brunnermunzel_one_sided(self):
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='less')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='greater')
        u3, p3 = stats.brunnermunzel(self.X, self.Y, alternative='greater')
        u4, p4 = stats.brunnermunzel(self.Y, self.X, alternative='less')
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(p3, p4, significant=self.significant)
        assert_(p1 != p3)
        assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(u3, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u4, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(p1, 0.002893104333075734, significant=self.significant)
        assert_approx_equal(p3, 0.9971068956669242, significant=self.significant)

    def test_brunnermunzel_two_sided(self):
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='two-sided')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='two-sided')
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(p1, 0.005786208666151538, significant=self.significant)

    def test_brunnermunzel_default(self):
        u1, p1 = stats.brunnermunzel(self.X, self.Y)
        u2, p2 = stats.brunnermunzel(self.Y, self.X)
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(p1, 0.005786208666151538, significant=self.significant)

    def test_brunnermunzel_alternative_error(self):
        alternative = 'error'
        distribution = 't'
        nan_policy = 'propagate'
        assert_(alternative not in ['two-sided', 'greater', 'less'])
        assert_raises(ValueError, stats.brunnermunzel, self.X, self.Y, alternative, distribution, nan_policy)

    def test_brunnermunzel_distribution_norm(self):
        u1, p1 = stats.brunnermunzel(self.X, self.Y, distribution='normal')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, distribution='normal')
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(p1, 0.0017041417600383024, significant=self.significant)

    def test_brunnermunzel_distribution_error(self):
        alternative = 'two-sided'
        distribution = 'error'
        nan_policy = 'propagate'
        assert_(alternative not in ['t', 'normal'])
        assert_raises(ValueError, stats.brunnermunzel, self.X, self.Y, alternative, distribution, nan_policy)

    def test_brunnermunzel_empty_imput(self):
        u1, p1 = stats.brunnermunzel(self.X, [])
        u2, p2 = stats.brunnermunzel([], self.Y)
        u3, p3 = stats.brunnermunzel([], [])
        assert_equal(u1, np.nan)
        assert_equal(p1, np.nan)
        assert_equal(u2, np.nan)
        assert_equal(p2, np.nan)
        assert_equal(u3, np.nan)
        assert_equal(p3, np.nan)

    def test_brunnermunzel_nan_input_propagate(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy='propagate')
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy='propagate')
        assert_equal(u1, np.nan)
        assert_equal(p1, np.nan)
        assert_equal(u2, np.nan)
        assert_equal(p2, np.nan)

    def test_brunnermunzel_nan_input_raise(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        alternative = 'two-sided'
        distribution = 't'
        nan_policy = 'raise'
        assert_raises(ValueError, stats.brunnermunzel, X, Y, alternative, distribution, nan_policy)
        assert_raises(ValueError, stats.brunnermunzel, Y, X, alternative, distribution, nan_policy)

    def test_brunnermunzel_nan_input_omit(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy='omit')
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy='omit')
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(u1, 3.1374674823029505, significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505, significant=self.significant)
        assert_approx_equal(p1, 0.005786208666151538, significant=self.significant)

    def test_brunnermunzel_return_nan(self):
        """ tests that a warning is emitted when p is nan
        p-value with t-distributions can be nan (0/0) (see gh-15843)
        """
        x = [1, 2, 3]
        y = [5, 6, 7, 8, 9]
        msg = 'p-value cannot be estimated|divide by zero|invalid value encountered'
        with pytest.warns(RuntimeWarning, match=msg):
            stats.brunnermunzel(x, y, distribution='t')

    def test_brunnermunzel_normal_dist(self):
        """ tests that a p is 0 for datasets that cause p->nan
        when t-distribution is used (see gh-15843)
        """
        x = [1, 2, 3]
        y = [5, 6, 7, 8, 9]
        with pytest.warns(RuntimeWarning, match='divide by zero'):
            _, p = stats.brunnermunzel(x, y, distribution='normal')
        assert_equal(p, 0)