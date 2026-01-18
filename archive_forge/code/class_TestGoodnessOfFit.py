import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
class TestGoodnessOfFit:

    def test_gof_iv(self):
        dist = stats.norm
        x = [1, 2, 3]
        message = '`dist` must be a \\(non-frozen\\) instance of...'
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(stats.norm(), x)
        message = '`data` must be a one-dimensional array of numbers.'
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, [[1, 2, 3]])
        message = '`statistic` must be one of...'
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, statistic='mm')
        message = '`n_mc_samples` must be an integer.'
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(dist, x, n_mc_samples=1000.5)
        message = "'herring' cannot be used to seed a"
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, random_state='herring')

    def test_against_ks(self):
        rng = np.random.default_rng(8517426291317196949)
        x = examgrades
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ks', random_state=rng)
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        assert_allclose(res.statistic, ref.statistic)
        assert_allclose(res.pvalue, ref.pvalue, atol=0.005)

    def test_against_lilliefors(self):
        rng = np.random.default_rng(2291803665717442724)
        x = examgrades
        res = goodness_of_fit(stats.norm, x, statistic='ks', random_state=rng)
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        assert_allclose(res.statistic, ref.statistic)
        assert_allclose(res.pvalue, 0.0348, atol=0.005)

    def test_against_cvm(self):
        rng = np.random.default_rng(8674330857509546614)
        x = examgrades
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='cvm', random_state=rng)
        ref = stats.cramervonmises(x, stats.norm(**known_params).cdf)
        assert_allclose(res.statistic, ref.statistic)
        assert_allclose(res.pvalue, ref.pvalue, atol=0.005)

    def test_against_anderson_case_0(self):
        rng = np.random.default_rng(7384539336846690410)
        x = np.arange(1, 101)
        known_params = {'loc': 45.01575354024957, 'scale': 30}
        res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 2.492)
        assert_allclose(res.pvalue, 0.05, atol=0.005)

    def test_against_anderson_case_1(self):
        rng = np.random.default_rng(5040212485680146248)
        x = np.arange(1, 101)
        known_params = {'scale': 29.957112639101933}
        res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 0.908)
        assert_allclose(res.pvalue, 0.1, atol=0.005)

    def test_against_anderson_case_2(self):
        rng = np.random.default_rng(726693985720914083)
        x = np.arange(1, 101)
        known_params = {'loc': 44.5680212261933}
        res = goodness_of_fit(stats.norm, x, known_params=known_params, statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 2.904)
        assert_allclose(res.pvalue, 0.025, atol=0.005)

    def test_against_anderson_case_3(self):
        rng = np.random.default_rng(6763691329830218206)
        x = stats.skewnorm.rvs(1.4477847789132101, loc=1, scale=2, size=100, random_state=rng)
        res = goodness_of_fit(stats.norm, x, statistic='ad', random_state=rng)
        assert_allclose(res.statistic, 0.559)
        assert_allclose(res.pvalue, 0.15, atol=0.005)

    @pytest.mark.slow
    def test_against_anderson_gumbel_r(self):
        rng = np.random.default_rng(7302761058217743)
        x = stats.genextreme(0.051896837188595134, loc=0.5, scale=1.5).rvs(size=1000, random_state=rng)
        res = goodness_of_fit(stats.gumbel_r, x, statistic='ad', random_state=rng)
        ref = stats.anderson(x, dist='gumbel_r')
        assert_allclose(res.statistic, ref.critical_values[0])
        assert_allclose(res.pvalue, ref.significance_level[0] / 100, atol=0.005)

    def test_against_filliben_norm(self):
        rng = np.random.default_rng(8024266430745011915)
        y = [6, 1, -4, 8, -2, 5, 0]
        known_params = {'loc': 0, 'scale': 1}
        res = stats.goodness_of_fit(stats.norm, y, known_params=known_params, statistic='filliben', random_state=rng)
        assert_allclose(res.statistic, 0.98538, atol=0.0001)
        assert 0.75 < res.pvalue < 0.9
        assert_allclose(res.statistic, 0.98540957187084, rtol=2e-05)
        assert_allclose(res.pvalue, 0.8875, rtol=0.002)

    def test_filliben_property(self):
        rng = np.random.default_rng(8535677809395478813)
        x = rng.normal(loc=10, scale=0.5, size=100)
        res = stats.goodness_of_fit(stats.norm, x, statistic='filliben', random_state=rng)
        known_params = {'loc': 0, 'scale': 1}
        ref = stats.goodness_of_fit(stats.norm, x, known_params=known_params, statistic='filliben', random_state=rng)
        assert_allclose(res.statistic, ref.statistic, rtol=1e-15)

    @pytest.mark.parametrize('case', [(25, [0.928, 0.937, 0.95, 0.958, 0.966]), (50, [0.959, 0.965, 0.972, 0.977, 0.981]), (95, [0.977, 0.979, 0.983, 0.986, 0.989])])
    def test_against_filliben_norm_table(self, case):
        rng = np.random.default_rng(504569995557928957)
        n, ref = case
        x = rng.random(n)
        known_params = {'loc': 0, 'scale': 1}
        res = stats.goodness_of_fit(stats.norm, x, known_params=known_params, statistic='filliben', random_state=rng)
        percentiles = np.array([0.005, 0.01, 0.025, 0.05, 0.1])
        res = stats.scoreatpercentile(res.null_distribution, percentiles * 100)
        assert_allclose(res, ref, atol=0.002)

    @pytest.mark.slow
    @pytest.mark.parametrize('case', [(5, 0.95772790260469, 0.4755), (6, 0.95398832257958, 0.3848), (7, 0.9432692889277, 0.2328)])
    def test_against_ppcc(self, case):
        n, ref_statistic, ref_pvalue = case
        rng = np.random.default_rng(7777775561439803116)
        x = rng.normal(size=n)
        res = stats.goodness_of_fit(stats.rayleigh, x, statistic='filliben', random_state=rng)
        assert_allclose(res.statistic, ref_statistic, rtol=0.0001)
        assert_allclose(res.pvalue, ref_pvalue, atol=0.015)

    def test_params_effects(self):
        rng = np.random.default_rng(9121950977643805391)
        x = stats.skewnorm.rvs(-5.044559778383153, loc=1, scale=2, size=50, random_state=rng)
        guessed_params = {'c': 13.4}
        fit_params = {'scale': 13.73}
        known_params = {'loc': -13.85}
        rng = np.random.default_rng(9121950977643805391)
        res1 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
        assert not np.allclose(res1.fit_result.params.c, 13.4)
        assert_equal(res1.fit_result.params.scale, 13.73)
        assert_equal(res1.fit_result.params.loc, -13.85)
        guessed_params = {'c': 2}
        rng = np.random.default_rng(9121950977643805391)
        res2 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
        assert not np.allclose(res2.fit_result.params.c, res1.fit_result.params.c, rtol=1e-08)
        assert not np.allclose(res2.null_distribution, res1.null_distribution, rtol=1e-08)
        assert_equal(res2.fit_result.params.scale, 13.73)
        assert_equal(res2.fit_result.params.loc, -13.85)
        fit_params = {'c': 13.4, 'scale': 13.73}
        rng = np.random.default_rng(9121950977643805391)
        res3 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2, guessed_params=guessed_params, fit_params=fit_params, known_params=known_params, random_state=rng)
        assert_equal(res3.fit_result.params.c, 13.4)
        assert_equal(res3.fit_result.params.scale, 13.73)
        assert_equal(res3.fit_result.params.loc, -13.85)
        assert not np.allclose(res3.null_distribution, res1.null_distribution)