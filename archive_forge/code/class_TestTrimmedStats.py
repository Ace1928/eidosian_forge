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
class TestTrimmedStats:
    dprec = np.finfo(np.float64).precision

    def test_tmean(self):
        y = stats.tmean(X, (2, 8), (True, True))
        assert_approx_equal(y, 5.0, significant=self.dprec)
        y1 = stats.tmean(X, limits=(2, 8), inclusive=(False, False))
        y2 = stats.tmean(X, limits=None)
        assert_approx_equal(y1, y2, significant=self.dprec)
        x_2d = arange(63, dtype=float64).reshape(9, 7)
        y = stats.tmean(x_2d, axis=None)
        assert_approx_equal(y, x_2d.mean(), significant=self.dprec)
        y = stats.tmean(x_2d, axis=0)
        assert_array_almost_equal(y, x_2d.mean(axis=0), decimal=8)
        y = stats.tmean(x_2d, axis=1)
        assert_array_almost_equal(y, x_2d.mean(axis=1), decimal=8)
        y = stats.tmean(x_2d, limits=(2, 61), axis=None)
        assert_approx_equal(y, 31.5, significant=self.dprec)
        y = stats.tmean(x_2d, limits=(2, 21), axis=0)
        y_true = [14, 11.5, 9, 10, 11, 12, 13]
        assert_array_almost_equal(y, y_true, decimal=8)
        y = stats.tmean(x_2d, limits=(2, 21), inclusive=(True, False), axis=0)
        y_true = [10.5, 11.5, 9, 10, 11, 12, 13]
        assert_array_almost_equal(y, y_true, decimal=8)
        x_2d_with_nan = np.array(x_2d)
        x_2d_with_nan[-1, -3:] = np.nan
        y = stats.tmean(x_2d_with_nan, limits=(1, 13), axis=0)
        y_true = [7, 4.5, 5.5, 6.5, np.nan, np.nan, np.nan]
        assert_array_almost_equal(y, y_true, decimal=8)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'Mean of empty slice')
            y = stats.tmean(x_2d, limits=(2, 21), axis=1)
            y_true = [4, 10, 17, 21, np.nan, np.nan, np.nan, np.nan, np.nan]
            assert_array_almost_equal(y, y_true, decimal=8)
            y = stats.tmean(x_2d, limits=(2, 21), inclusive=(False, True), axis=1)
            y_true = [4.5, 10, 17, 21, np.nan, np.nan, np.nan, np.nan, np.nan]
            assert_array_almost_equal(y, y_true, decimal=8)

    def test_tvar(self):
        y = stats.tvar(X, limits=(2, 8), inclusive=(True, True))
        assert_approx_equal(y, 4.666666666666666, significant=self.dprec)
        y = stats.tvar(X, limits=None)
        assert_approx_equal(y, X.var(ddof=1), significant=self.dprec)
        x_2d = arange(63, dtype=float64).reshape((9, 7))
        y = stats.tvar(x_2d, axis=None)
        assert_approx_equal(y, x_2d.var(ddof=1), significant=self.dprec)
        y = stats.tvar(x_2d, axis=0)
        assert_array_almost_equal(y[0], np.full((1, 7), 367.5), decimal=8)
        y = stats.tvar(x_2d, axis=1)
        assert_array_almost_equal(y[0], np.full((1, 9), 4.66666667), decimal=8)
        y = stats.tvar(x_2d[3, :])
        assert_approx_equal(y, 4.666666666666667, significant=self.dprec)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'Degrees of freedom <= 0 for slice.')
            y = stats.tvar(x_2d, limits=(1, 5), axis=1, inclusive=(True, True))
            assert_approx_equal(y[0], 2.5, significant=self.dprec)
            y = stats.tvar(x_2d, limits=(0, 6), axis=1, inclusive=(True, True))
            assert_approx_equal(y[0], 4.666666666666667, significant=self.dprec)
            assert_equal(y[1], np.nan)

    def test_tstd(self):
        y = stats.tstd(X, (2, 8), (True, True))
        assert_approx_equal(y, 2.1602468994692865, significant=self.dprec)
        y = stats.tstd(X, limits=None)
        assert_approx_equal(y, X.std(ddof=1), significant=self.dprec)

    def test_tmin(self):
        assert_equal(stats.tmin(4), 4)
        x = np.arange(10)
        assert_equal(stats.tmin(x), 0)
        assert_equal(stats.tmin(x, lowerlimit=0), 0)
        assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), 1)
        x = x.reshape((5, 2))
        assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), [2, 1])
        assert_equal(stats.tmin(x, axis=1), [0, 2, 4, 6, 8])
        assert_equal(stats.tmin(x, axis=None), 0)
        x = np.arange(10.0)
        x[9] = np.nan
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'invalid value*')
            assert_equal(stats.tmin(x), np.nan)
            assert_equal(stats.tmin(x, nan_policy='omit'), 0.0)
            assert_raises(ValueError, stats.tmin, x, nan_policy='raise')
            assert_raises(ValueError, stats.tmin, x, nan_policy='foobar')
            msg = "'propagate', 'raise', 'omit'"
            with assert_raises(ValueError, match=msg):
                stats.tmin(x, nan_policy='foo')
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'All-NaN slice encountered')
            x = np.arange(16).reshape(4, 4)
            res = stats.tmin(x, lowerlimit=4, axis=1)
            assert_equal(res, [np.nan, 4, 8, 12])

    def test_tmax(self):
        assert_equal(stats.tmax(4), 4)
        x = np.arange(10)
        assert_equal(stats.tmax(x), 9)
        assert_equal(stats.tmax(x, upperlimit=9), 9)
        assert_equal(stats.tmax(x, upperlimit=9, inclusive=False), 8)
        x = x.reshape((5, 2))
        assert_equal(stats.tmax(x, upperlimit=9, inclusive=False), [8, 7])
        assert_equal(stats.tmax(x, axis=1), [1, 3, 5, 7, 9])
        assert_equal(stats.tmax(x, axis=None), 9)
        x = np.arange(10.0)
        x[6] = np.nan
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'invalid value*')
            assert_equal(stats.tmax(x), np.nan)
            assert_equal(stats.tmax(x, nan_policy='omit'), 9.0)
            assert_raises(ValueError, stats.tmax, x, nan_policy='raise')
            assert_raises(ValueError, stats.tmax, x, nan_policy='foobar')
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'All-NaN slice encountered')
            x = np.arange(16).reshape(4, 4)
            res = stats.tmax(x, upperlimit=11, axis=1)
            assert_equal(res, [3, 7, 11, np.nan])

    def test_tsem(self):
        y = stats.tsem(X, limits=(3, 8), inclusive=(False, True))
        y_ref = np.array([4, 5, 6, 7, 8])
        assert_approx_equal(y, y_ref.std(ddof=1) / np.sqrt(y_ref.size), significant=self.dprec)
        assert_approx_equal(stats.tsem(X, limits=[-1, 10]), stats.tsem(X, limits=None), significant=self.dprec)