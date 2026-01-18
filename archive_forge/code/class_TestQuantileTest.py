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
class TestQuantileTest:
    """ Test the non-parametric quantile test,
    including the computation of confidence intervals
    """

    def test_quantile_test_iv(self):
        x = [1, 2, 3]
        message = '`x` must be a one-dimensional array of numbers.'
        with pytest.raises(ValueError, match=message):
            stats.quantile_test([x])
        message = '`q` must be a scalar.'
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, q=[1, 2])
        message = '`p` must be a float strictly between 0 and 1.'
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=[0.5, 0.75])
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=2)
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=-0.5)
        message = '`alternative` must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, alternative='one-sided')
        message = '`confidence_level` must be a number between 0 and 1.'
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x).confidence_interval(1)

    @pytest.mark.parametrize('p, alpha, lb, ub, alternative', [[0.3, 0.95, 1.22140275816017, 1.476980793882643, 'two-sided'], [0.5, 0.9, 1.506817785112854, 1.803988415397857, 'two-sided'], [0.25, 0.95, -np.inf, 1.39096812846378, 'less'], [0.8, 0.9, 2.117000016612675, np.inf, 'greater']])
    def test_R_ci_quantile(self, p, alpha, lb, ub, alternative):
        x = np.exp(np.arange(0, 1.01, 0.01))
        res = stats.quantile_test(x, p=p, alternative=alternative)
        assert_allclose(res.confidence_interval(alpha), [lb, ub], rtol=1e-15)

    @pytest.mark.parametrize('q, p, alternative, ref', [[1.2, 0.3, 'two-sided', 0.01515567517648], [1.8, 0.5, 'two-sided', 0.1109183496606]])
    def test_R_pvalue(self, q, p, alternative, ref):
        x = np.exp(np.arange(0, 1.01, 0.01))
        res = stats.quantile_test(x, q=q, p=p, alternative=alternative)
        assert_allclose(res.pvalue, ref, rtol=1e-12)

    @pytest.mark.parametrize('case', ['continuous', 'discrete'])
    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    @pytest.mark.parametrize('alpha', [0.9, 0.95])
    def test_pval_ci_match(self, case, alternative, alpha):
        seed = int((7 ** len(case) + len(alternative)) * alpha)
        rng = np.random.default_rng(seed)
        if case == 'continuous':
            p, q = rng.random(size=2)
            rvs = rng.random(size=100)
        else:
            rvs = rng.integers(1, 11, size=100)
            p = rng.random()
            q = rng.integers(1, 11)
        res = stats.quantile_test(rvs, q=q, p=p, alternative=alternative)
        ci = res.confidence_interval(confidence_level=alpha)
        if alternative == 'less':
            i_inside = rvs <= ci.high
        else:
            i_inside = rvs >= ci.low
        for x in rvs[i_inside]:
            res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
            assert res.pvalue > 1 - alpha
        for x in rvs[~i_inside]:
            res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
            assert res.pvalue < 1 - alpha

    def test_match_conover_examples(self):
        x = [189, 233, 195, 160, 212, 176, 231, 185, 199, 213, 202, 193, 174, 166, 248]
        pvalue_expected = 0.0346
        res = stats.quantile_test(x, q=193, p=0.75, alternative='two-sided')
        assert_allclose(res.pvalue, pvalue_expected, rtol=1e-05)
        x = [59] * 8 + [61] * (112 - 8)
        pvalue_expected = stats.binom(p=0.5, n=112).pmf(k=8)
        res = stats.quantile_test(x, q=60, p=0.5, alternative='greater')
        assert_allclose(res.pvalue, pvalue_expected, atol=1e-10)