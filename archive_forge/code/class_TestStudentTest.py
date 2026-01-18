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
class TestStudentTest:
    X1 = np.array([-1, 0, 1])
    X2 = np.array([0, 1, 2])
    T1_0 = 0
    P1_0 = 1
    T1_1 = -1.7320508075
    P1_1 = 0.22540333075
    T1_2 = -3.464102
    P1_2 = 0.0741799
    T2_0 = 1.732051
    P2_0 = 0.2254033
    P1_1_l = P1_1 / 2
    P1_1_g = 1 - P1_1 / 2

    def test_onesample(self):
        with suppress_warnings() as sup, np.errstate(invalid='ignore', divide='ignore'):
            sup.filter(RuntimeWarning, 'Degrees of freedom <= 0 for slice')
            t, p = stats.ttest_1samp(4.0, 3.0)
        assert_(np.isnan(t))
        assert_(np.isnan(p))
        t, p = stats.ttest_1samp(self.X1, 0)
        assert_array_almost_equal(t, self.T1_0)
        assert_array_almost_equal(p, self.P1_0)
        res = stats.ttest_1samp(self.X1, 0)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)
        t, p = stats.ttest_1samp(self.X2, 0)
        assert_array_almost_equal(t, self.T2_0)
        assert_array_almost_equal(p, self.P2_0)
        t, p = stats.ttest_1samp(self.X1, 1)
        assert_array_almost_equal(t, self.T1_1)
        assert_array_almost_equal(p, self.P1_1)
        t, p = stats.ttest_1samp(self.X1, 2)
        assert_array_almost_equal(t, self.T1_2)
        assert_array_almost_equal(p, self.P1_2)
        x = stats.norm.rvs(loc=5, scale=10, size=51, random_state=7654567)
        x[50] = np.nan
        with np.errstate(invalid='ignore'):
            assert_array_equal(stats.ttest_1samp(x, 5.0), (np.nan, np.nan))
            assert_array_almost_equal(stats.ttest_1samp(x, 5.0, nan_policy='omit'), (-1.641262407436716, 0.107147027334048))
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='raise')
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='foobar')

    def test_1samp_alternative(self):
        assert_raises(ValueError, stats.ttest_1samp, self.X1, 0, alternative='error')
        t, p = stats.ttest_1samp(self.X1, 1, alternative='less')
        assert_allclose(p, self.P1_1_l)
        assert_allclose(t, self.T1_1)
        t, p = stats.ttest_1samp(self.X1, 1, alternative='greater')
        assert_allclose(p, self.P1_1_g)
        assert_allclose(t, self.T1_1)

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    def test_1samp_ci_1d(self, alternative):
        rng = np.random.default_rng(8066178009154342972)
        n = 10
        x = rng.normal(size=n, loc=1.5, scale=2)
        popmean = rng.normal()
        ref = {'two-sided': [0.3594423211709136, 2.933345502829086], 'greater': [0.7470806207371626, np.inf], 'less': [-np.inf, 2.545707203262837]}
        res = stats.ttest_1samp(x, popmean=popmean, alternative=alternative)
        ci = res.confidence_interval(confidence_level=0.85)
        assert_allclose(ci, ref[alternative])
        assert_equal(res.df, n - 1)

    def test_1samp_ci_iv(self):
        res = stats.ttest_1samp(np.arange(10), 0)
        message = '`confidence_level` must be a number between 0 and 1.'
        with pytest.raises(ValueError, match=message):
            res.confidence_interval(confidence_level=10)

    @pytest.mark.xslow
    @hypothesis.given(alpha=hypothesis.strategies.floats(1e-15, 1 - 1e-15), data_axis=ttest_data_axis_strategy())
    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    def test_pvalue_ci(self, alpha, data_axis, alternative):
        data, axis = data_axis
        res = stats.ttest_1samp(data, 0, alternative=alternative, axis=axis)
        l, u = res.confidence_interval(confidence_level=alpha)
        popmean = l if alternative == 'greater' else u
        popmean = np.expand_dims(popmean, axis=axis)
        res = stats.ttest_1samp(data, popmean, alternative=alternative, axis=axis)
        np.testing.assert_allclose(res.pvalue, 1 - alpha)