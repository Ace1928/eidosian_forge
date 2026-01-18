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
class Test_ttest_ind_permutations:
    N = 20
    np.random.seed(0)
    a = np.vstack((np.arange(3 * N // 4), np.random.random(3 * N // 4)))
    b = np.vstack((np.arange(N // 4) + 100, np.random.random(N // 4)))
    a2 = np.arange(10)
    b2 = np.arange(10) + 100
    a3 = [1, 2]
    b3 = [3, 4]
    np.random.seed(0)
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500).reshape(100, 5).T
    rvs2 = stats.norm.rvs(loc=8, scale=20, size=100)
    p_d = [1 / 1001, (676 + 1) / 1001]
    p_d_gen = [1 / 1001, (672 + 1) / 1001]
    p_d_big = [(993 + 1) / 1001, (685 + 1) / 1001, (840 + 1) / 1001, (955 + 1) / 1001, (255 + 1) / 1001]
    params = [(a, b, {'axis': 1}, p_d), (a.T, b.T, {'axis': 0}, p_d), (a[0, :], b[0, :], {'axis': None}, p_d[0]), (a[0, :].tolist(), b[0, :].tolist(), {'axis': None}, p_d[0]), (a, b, {'random_state': 0, 'axis': 1}, p_d), (a, b, {'random_state': np.random.RandomState(0), 'axis': 1}, p_d), (a2, b2, {'equal_var': True}, 1 / 1001), (rvs1, rvs2, {'axis': -1, 'random_state': 0}, p_d_big), (a3, b3, {}, 1 / 3), (a, b, {'random_state': np.random.default_rng(0), 'axis': 1}, p_d_gen)]

    @pytest.mark.parametrize('a,b,update,p_d', params)
    def test_ttest_ind_permutations(self, a, b, update, p_d):
        options_a = {'axis': None, 'equal_var': False}
        options_p = {'axis': None, 'equal_var': False, 'permutations': 1000, 'random_state': 0}
        options_a.update(update)
        options_p.update(update)
        stat_a, _ = stats.ttest_ind(a, b, **options_a)
        stat_p, pvalue = stats.ttest_ind(a, b, **options_p)
        assert_array_almost_equal(stat_a, stat_p, 5)
        assert_array_almost_equal(pvalue, p_d)

    def test_ttest_ind_exact_alternative(self):
        np.random.seed(0)
        N = 3
        a = np.random.rand(2, N, 2)
        b = np.random.rand(2, N, 2)
        options_p = {'axis': 1, 'permutations': 1000}
        options_p.update(alternative='greater')
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        res_g_ba = stats.ttest_ind(b, a, **options_p)
        options_p.update(alternative='less')
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        res_l_ba = stats.ttest_ind(b, a, **options_p)
        options_p.update(alternative='two-sided')
        res_2_ab = stats.ttest_ind(a, b, **options_p)
        res_2_ba = stats.ttest_ind(b, a, **options_p)
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)
        assert_equal(res_g_ab.statistic, res_2_ab.statistic)
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)
        assert_equal(res_2_ab.statistic, -res_2_ba.statistic)
        assert_equal(res_2_ab.pvalue, res_2_ba.pvalue)
        assert_equal(res_g_ab.pvalue, res_l_ba.pvalue)
        assert_equal(res_l_ab.pvalue, res_g_ba.pvalue)
        mask = res_g_ab.pvalue <= 0.5
        assert_equal(res_g_ab.pvalue[mask] + res_l_ba.pvalue[mask], res_2_ab.pvalue[mask])
        assert_equal(res_l_ab.pvalue[~mask] + res_g_ba.pvalue[~mask], res_2_ab.pvalue[~mask])

    def test_ttest_ind_exact_selection(self):
        np.random.seed(0)
        N = 3
        a = np.random.rand(N)
        b = np.random.rand(N)
        res0 = stats.ttest_ind(a, b)
        res1 = stats.ttest_ind(a, b, permutations=1000)
        res2 = stats.ttest_ind(a, b, permutations=0)
        res3 = stats.ttest_ind(a, b, permutations=np.inf)
        assert res1.pvalue != res0.pvalue
        assert res2.pvalue == res0.pvalue
        assert res3.pvalue == res1.pvalue

    def test_ttest_ind_exact_distribution(self):
        np.random.seed(0)
        a = np.random.rand(3)
        b = np.random.rand(4)
        data = np.concatenate((a, b))
        na, nb = (len(a), len(b))
        permutations = 100000
        t_stat, _, _ = _permutation_distribution_t(data, permutations, na, True)
        n_unique = len(set(t_stat))
        assert n_unique == binom(na + nb, na)
        assert len(t_stat) == n_unique

    def test_ttest_ind_randperm_alternative(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(2, 3, N)
        b = np.random.rand(3, N)
        options_p = {'axis': -1, 'permutations': 1000, 'random_state': 0}
        options_p.update(alternative='greater')
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        res_g_ba = stats.ttest_ind(b, a, **options_p)
        options_p.update(alternative='less')
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        res_l_ba = stats.ttest_ind(b, a, **options_p)
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1 + 1 / (options_p['permutations'] + 1))
        assert_equal(res_g_ba.pvalue + res_l_ba.pvalue, 1 + 1 / (options_p['permutations'] + 1))

    @pytest.mark.slow()
    def test_ttest_ind_randperm_alternative2(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(N, 4)
        b = np.random.rand(N, 4)
        options_p = {'permutations': 20000, 'random_state': 0}
        options_p.update(alternative='greater')
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        options_p.update(alternative='less')
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        options_p.update(alternative='two-sided')
        res_2_ab = stats.ttest_ind(a, b, **options_p)
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue, 1 + 1 / (options_p['permutations'] + 1))
        mask = res_g_ab.pvalue <= 0.5
        assert_allclose(2 * res_g_ab.pvalue[mask], res_2_ab.pvalue[mask], atol=0.02)
        assert_allclose(2 * (1 - res_g_ab.pvalue[~mask]), res_2_ab.pvalue[~mask], atol=0.02)
        assert_allclose(2 * res_l_ab.pvalue[~mask], res_2_ab.pvalue[~mask], atol=0.02)
        assert_allclose(2 * (1 - res_l_ab.pvalue[mask]), res_2_ab.pvalue[mask], atol=0.02)

    def test_ttest_ind_permutation_nanpolicy(self):
        np.random.seed(0)
        N = 50
        a = np.random.rand(N, 5)
        b = np.random.rand(N, 5)
        a[5, 1] = np.nan
        b[8, 2] = np.nan
        a[9, 3] = np.nan
        b[9, 3] = np.nan
        options_p = {'permutations': 1000, 'random_state': 0}
        options_p.update(nan_policy='raise')
        with assert_raises(ValueError, match='The input contains nan values'):
            res = stats.ttest_ind(a, b, **options_p)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, 'invalid value*')
            options_p.update(nan_policy='propagate')
            res = stats.ttest_ind(a, b, **options_p)
            mask = np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0)
            res2 = stats.ttest_ind(a[:, ~mask], b[:, ~mask], **options_p)
            assert_equal(res.pvalue[mask], np.nan)
            assert_equal(res.statistic[mask], np.nan)
            assert_allclose(res.pvalue[~mask], res2.pvalue)
            assert_allclose(res.statistic[~mask], res2.statistic)
            res = stats.ttest_ind(a.ravel(), b.ravel(), **options_p)
            assert np.isnan(res.pvalue)
            assert np.isnan(res.statistic)

    def test_ttest_ind_permutation_check_inputs(self):
        with assert_raises(ValueError, match='Permutations must be'):
            stats.ttest_ind(self.a2, self.b2, permutations=-3)
        with assert_raises(ValueError, match='Permutations must be'):
            stats.ttest_ind(self.a2, self.b2, permutations=1.5)
        with assert_raises(ValueError, match="'hello' cannot be used"):
            stats.ttest_ind(self.a, self.b, permutations=1, random_state='hello', axis=1)

    def test_ttest_ind_permutation_check_p_values(self):
        N = 10
        a = np.random.rand(N, 20)
        b = np.random.rand(N, 20)
        p_values = stats.ttest_ind(a, b, permutations=1).pvalue
        print(0.0 not in p_values)
        assert 0.0 not in p_values