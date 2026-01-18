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
class TestMoments:
    """
        Comparison numbers are found using R v.1.5.1
        note that length(testcase) = 4
        testmathworks comes from documentation for the
        Statistics Toolbox for Matlab and can be found at both
        https://www.mathworks.com/help/stats/kurtosis.html
        https://www.mathworks.com/help/stats/skewness.html
        Note that both test cases came from here.
    """
    testcase = [1, 2, 3, 4]
    scalar_testcase = 4.0
    np.random.seed(1234)
    testcase_moment_accuracy = np.random.rand(42)
    testmathworks = [1.165, 0.6268, 0.0751, 0.3516, -0.6965]

    def _assert_equal(self, actual, expect, *, shape=None, dtype=None):
        expect = np.asarray(expect)
        if shape is not None:
            expect = np.broadcast_to(expect, shape)
        assert_array_equal(actual, expect)
        if dtype is None:
            dtype = expect.dtype
        assert actual.dtype == dtype

    @pytest.mark.parametrize('size', [10, (10, 2)])
    @pytest.mark.parametrize('m, c', product((0, 1, 2, 3), (None, 0, 1)))
    def test_moment_center_scalar_moment(self, size, m, c):
        rng = np.random.default_rng(6581432544381372042)
        x = rng.random(size=size)
        res = stats.moment(x, m, center=c)
        c = np.mean(x, axis=0) if c is None else c
        ref = np.sum((x - c) ** m, axis=0) / len(x)
        assert_allclose(res, ref, atol=1e-16)

    @pytest.mark.parametrize('size', [10, (10, 2)])
    @pytest.mark.parametrize('c', (None, 0, 1))
    def test_moment_center_array_moment(self, size, c):
        rng = np.random.default_rng(1706828300224046506)
        x = rng.random(size=size)
        m = [0, 1, 2, 3]
        res = stats.moment(x, m, center=c)
        ref = [stats.moment(x, i, center=c) for i in m]
        assert_equal(res, ref)

    def test_moment(self):
        y = stats.moment(self.scalar_testcase)
        assert_approx_equal(y, 0.0)
        y = stats.moment(self.testcase, 0)
        assert_approx_equal(y, 1.0)
        y = stats.moment(self.testcase, 1)
        assert_approx_equal(y, 0.0, 10)
        y = stats.moment(self.testcase, 2)
        assert_approx_equal(y, 1.25)
        y = stats.moment(self.testcase, 3)
        assert_approx_equal(y, 0.0)
        y = stats.moment(self.testcase, 4)
        assert_approx_equal(y, 2.5625)
        y = stats.moment(self.testcase, [1, 2, 3, 4])
        assert_allclose(y, [0, 1.25, 0, 2.5625])
        y = stats.moment(self.testcase, 0.0)
        assert_approx_equal(y, 1.0)
        assert_raises(ValueError, stats.moment, self.testcase, 1.2)
        y = stats.moment(self.testcase, [1.0, 2, 3, 4.0])
        assert_allclose(y, [0, 1.25, 0, 2.5625])
        message = 'Mean of empty slice\\.|invalid value encountered.*'
        with pytest.warns(RuntimeWarning, match=message):
            y = stats.moment([])
            self._assert_equal(y, np.nan, dtype=np.float64)
            y = stats.moment(np.array([], dtype=np.float32))
            self._assert_equal(y, np.nan, dtype=np.float32)
            y = stats.moment(np.zeros((1, 0)), axis=0)
            self._assert_equal(y, [], shape=(0,), dtype=np.float64)
            y = stats.moment([[]], axis=1)
            self._assert_equal(y, np.nan, shape=(1,), dtype=np.float64)
            y = stats.moment([[]], moment=[0, 1], axis=0)
            self._assert_equal(y, [], shape=(2, 0))
        x = np.arange(10.0)
        x[9] = np.nan
        assert_equal(stats.moment(x, 2), np.nan)
        assert_almost_equal(stats.moment(x, nan_policy='omit'), 0.0)
        assert_raises(ValueError, stats.moment, x, nan_policy='raise')
        assert_raises(ValueError, stats.moment, x, nan_policy='foobar')

    @pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex128])
    @pytest.mark.parametrize('expect, moment', [(0, 1), (1, 0)])
    def test_constant_moments(self, dtype, expect, moment):
        x = np.random.rand(5).astype(dtype)
        y = stats.moment(x, moment=moment)
        self._assert_equal(y, expect, dtype=dtype)
        y = stats.moment(np.broadcast_to(x, (6, 5)), axis=0, moment=moment)
        self._assert_equal(y, expect, shape=(5,), dtype=dtype)
        y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=2, moment=moment)
        self._assert_equal(y, expect, shape=(1, 2, 4, 5), dtype=dtype)
        y = stats.moment(np.broadcast_to(x, (1, 2, 3, 4, 5)), axis=None, moment=moment)
        self._assert_equal(y, expect, shape=(), dtype=dtype)

    def test_moment_propagate_nan(self):
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        mm = stats.moment(a, 2, axis=1, nan_policy='propagate')
        np.testing.assert_allclose(mm, [1.25, np.nan], atol=1e-15)

    def test_moment_empty_moment(self):
        with pytest.raises(ValueError, match="'moment' must be a scalar or a non-empty 1D list/array."):
            stats.moment([1, 2, 3, 4], moment=[])

    def test_skewness(self):
        y = stats.skew(self.scalar_testcase)
        assert np.isnan(y)
        y = stats.skew(self.testmathworks)
        assert_approx_equal(y, -0.29322304336607, 10)
        y = stats.skew(self.testmathworks, bias=0)
        assert_approx_equal(y, -0.43711110502394, 10)
        y = stats.skew(self.testcase)
        assert_approx_equal(y, 0.0, 10)
        x = np.arange(10.0)
        x[9] = np.nan
        with np.errstate(invalid='ignore'):
            assert_equal(stats.skew(x), np.nan)
        assert_equal(stats.skew(x, nan_policy='omit'), 0.0)
        assert_raises(ValueError, stats.skew, x, nan_policy='raise')
        assert_raises(ValueError, stats.skew, x, nan_policy='foobar')

    def test_skewness_scalar(self):
        assert_equal(stats.skew(arange(10)), 0.0)

    def test_skew_propagate_nan(self):
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        with np.errstate(invalid='ignore'):
            s = stats.skew(a, axis=1, nan_policy='propagate')
        np.testing.assert_allclose(s, [0, np.nan], atol=1e-15)

    def test_skew_constant_value(self):
        with pytest.warns(RuntimeWarning, match='Precision loss occurred'):
            a = np.repeat(-0.27829495, 10)
            assert np.isnan(stats.skew(a))
            assert np.isnan(stats.skew(a * float(2 ** 50)))
            assert np.isnan(stats.skew(a / float(2 ** 50)))
            assert np.isnan(stats.skew(a, bias=False))
            assert np.isnan(stats.skew([14.3] * 7))
            assert np.isnan(stats.skew(1 + np.arange(-3, 4) * 1e-16))

    def test_kurtosis(self):
        y = stats.kurtosis(self.scalar_testcase)
        assert np.isnan(y)
        y = stats.kurtosis(self.testmathworks, 0, fisher=0, bias=1)
        assert_approx_equal(y, 2.1658856802973, 10)
        y = stats.kurtosis(self.testmathworks, fisher=0, bias=0)
        assert_approx_equal(y, 3.663542721189047, 10)
        y = stats.kurtosis(self.testcase, 0, 0)
        assert_approx_equal(y, 1.64)
        x = np.arange(10.0)
        x[9] = np.nan
        assert_equal(stats.kurtosis(x), np.nan)
        assert_almost_equal(stats.kurtosis(x, nan_policy='omit'), -1.23)
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='raise')
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='foobar')

    def test_kurtosis_array_scalar(self):
        assert_equal(type(stats.kurtosis([1, 2, 3])), np.float64)

    def test_kurtosis_propagate_nan(self):
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        k = stats.kurtosis(a, axis=1, nan_policy='propagate')
        np.testing.assert_allclose(k, [-1.36, np.nan], atol=1e-15)

    def test_kurtosis_constant_value(self):
        a = np.repeat(-0.27829495, 10)
        with pytest.warns(RuntimeWarning, match='Precision loss occurred'):
            assert np.isnan(stats.kurtosis(a, fisher=False))
            assert np.isnan(stats.kurtosis(a * float(2 ** 50), fisher=False))
            assert np.isnan(stats.kurtosis(a / float(2 ** 50), fisher=False))
            assert np.isnan(stats.kurtosis(a, fisher=False, bias=False))

    def test_moment_accuracy(self):
        tc_no_mean = self.testcase_moment_accuracy - np.mean(self.testcase_moment_accuracy)
        assert_allclose(np.power(tc_no_mean, 42).mean(), stats.moment(self.testcase_moment_accuracy, 42))

    def test_precision_loss_gh15554(self):
        with pytest.warns(RuntimeWarning, match='Precision loss occurred'):
            rng = np.random.default_rng(34095309370)
            a = rng.random(size=(100, 10))
            a[:, 0] = 1.01
            stats.skew(a)[0]

    def test_empty_1d(self):
        message = 'Mean of empty slice\\.|invalid value encountered.*'
        with pytest.warns(RuntimeWarning, match=message):
            stats.skew([])
        with pytest.warns(RuntimeWarning, match=message):
            stats.kurtosis([])