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
class TestMGCStat:
    """ Test validity of MGC test statistic
    """

    def _simulations(self, samps=100, dims=1, sim_type=''):
        if sim_type == 'linear':
            x = np.random.uniform(-1, 1, size=(samps, 1))
            y = x + 0.3 * np.random.random_sample(size=(x.size, 1))
        elif sim_type == 'nonlinear':
            unif = np.array(np.random.uniform(0, 5, size=(samps, 1)))
            x = unif * np.cos(np.pi * unif)
            y = unif * np.sin(np.pi * unif) + 0.4 * np.random.random_sample(size=(x.size, 1))
        elif sim_type == 'independence':
            u = np.random.normal(0, 1, size=(samps, 1))
            v = np.random.normal(0, 1, size=(samps, 1))
            u_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            v_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            x = u / 3 + 2 * u_2 - 1
            y = v / 3 + 2 * v_2 - 1
        else:
            raise ValueError('sim_type must be linear, nonlinear, or independence')
        if dims > 1:
            dims_noise = np.random.normal(0, 1, size=(samps, dims - 1))
            x = np.concatenate((x, dims_noise), axis=1)
        return (x, y)

    @pytest.mark.slow
    @pytest.mark.parametrize('sim_type, obs_stat, obs_pvalue', [('linear', 0.97, 1 / 1000), ('nonlinear', 0.163, 1 / 1000), ('independence', -0.0094, 0.78)])
    def test_oned(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=1, sim_type=sim_type)
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.slow
    @pytest.mark.parametrize('sim_type, obs_stat, obs_pvalue', [('linear', 0.184, 1 / 1000), ('nonlinear', 0.019, 0.117)])
    def test_fived(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=5, sim_type=sim_type)
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.xslow
    def test_twosamp(self):
        np.random.seed(12345678)
        x = np.random.binomial(100, 0.5, size=(100, 5))
        y = np.random.normal(0, 1, size=(80, 5))
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, 1.0, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)
        y = np.random.normal(0, 1, size=(100, 5))
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, is_twosamp=True)
        assert_approx_equal(stat, 1.0, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_workers(self):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=1, sim_type='linear')
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, workers=2)
        assert_approx_equal(stat, 0.97, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_random_state(self):
        x, y = self._simulations(samps=100, dims=1, sim_type='linear')
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, random_state=1)
        assert_approx_equal(stat, 0.97, significant=1)
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.slow
    def test_dist_perm(self):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=1, sim_type='nonlinear')
        distx = cdist(x, x, metric='euclidean')
        disty = cdist(y, y, metric='euclidean')
        stat_dist, pvalue_dist, _ = stats.multiscale_graphcorr(distx, disty, compute_distance=None, random_state=1)
        assert_approx_equal(stat_dist, 0.163, significant=1)
        assert_approx_equal(pvalue_dist, 0.001, significant=1)

    @pytest.mark.slow
    def test_pvalue_literature(self):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=1, sim_type='linear')
        _, pvalue, _ = stats.multiscale_graphcorr(x, y, random_state=1)
        assert_allclose(pvalue, 1 / 1001)

    @pytest.mark.slow
    def test_alias(self):
        np.random.seed(12345678)
        x, y = self._simulations(samps=100, dims=1, sim_type='linear')
        res = stats.multiscale_graphcorr(x, y, random_state=1)
        assert_equal(res.stat, res.statistic)