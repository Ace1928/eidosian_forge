import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
class TestMonteCarloHypothesisTest:
    atol = 0.025

    def rvs(self, rvs_in, rs):
        return lambda *args, **kwds: rvs_in(*args, random_state=rs, **kwds)

    def test_input_validation(self):

        def stat(x):
            return stats.skewnorm(x).statistic
        message = 'Array shapes are incompatible for broadcasting.'
        data = (np.zeros((2, 5)), np.zeros((3, 5)))
        rvs = (stats.norm.rvs, stats.norm.rvs)
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, rvs, lambda x, y: 1, axis=-1)
        message = '`axis` must be an integer.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, axis=1.5)
        message = '`vectorized` must be `True`, `False`, or `None`.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, vectorized=1.5)
        message = '`rvs` must be callable or sequence of callables.'
        with pytest.raises(TypeError, match=message):
            monte_carlo_test([1, 2, 3], None, stat)
        with pytest.raises(TypeError, match=message):
            monte_carlo_test([[1, 2], [3, 4]], [lambda x: x, None], stat)
        message = 'If `rvs` is a sequence...'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([[1, 2, 3]], [lambda x: x, lambda x: x], stat)
        message = '`statistic` must be callable.'
        with pytest.raises(TypeError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, None)
        message = '`n_resamples` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, n_resamples=-1000)
        message = '`n_resamples` must be a positive integer.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, n_resamples=1000.5)
        message = '`batch` must be a positive integer or None.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, batch=-1000)
        message = '`batch` must be a positive integer or None.'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, batch=1000.5)
        message = '`alternative` must be in...'
        with pytest.raises(ValueError, match=message):
            monte_carlo_test([1, 2, 3], stats.norm.rvs, stat, alternative='ekki')

    def test_batch(self):
        rng = np.random.default_rng(23492340193)
        x = rng.random(10)

        def statistic(x, axis):
            batch_size = 1 if x.ndim == 1 else len(x)
            statistic.batch_size = max(batch_size, statistic.batch_size)
            statistic.counter += 1
            return stats.skewtest(x, axis=axis).statistic
        statistic.counter = 0
        statistic.batch_size = 0
        kwds = {'sample': x, 'statistic': statistic, 'n_resamples': 1000, 'vectorized': True}
        kwds['rvs'] = self.rvs(stats.norm.rvs, np.random.default_rng(32842398))
        res1 = monte_carlo_test(batch=1, **kwds)
        assert_equal(statistic.counter, 1001)
        assert_equal(statistic.batch_size, 1)
        kwds['rvs'] = self.rvs(stats.norm.rvs, np.random.default_rng(32842398))
        statistic.counter = 0
        res2 = monte_carlo_test(batch=50, **kwds)
        assert_equal(statistic.counter, 21)
        assert_equal(statistic.batch_size, 50)
        kwds['rvs'] = self.rvs(stats.norm.rvs, np.random.default_rng(32842398))
        statistic.counter = 0
        res3 = monte_carlo_test(**kwds)
        assert_equal(statistic.counter, 2)
        assert_equal(statistic.batch_size, 1000)
        assert_equal(res1.pvalue, res3.pvalue)
        assert_equal(res2.pvalue, res3.pvalue)

    @pytest.mark.parametrize('axis', range(-3, 3))
    def test_axis(self, axis):
        rng = np.random.default_rng(2389234)
        norm_rvs = self.rvs(stats.norm.rvs, rng)
        size = [2, 3, 4]
        size[axis] = 100
        x = norm_rvs(size=size)
        expected = stats.skewtest(x, axis=axis)

        def statistic(x, axis):
            return stats.skewtest(x, axis=axis).statistic
        res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True, n_resamples=20000, axis=axis)
        assert_allclose(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('alternative', ('less', 'greater'))
    @pytest.mark.parametrize('a', np.linspace(-0.5, 0.5, 5))
    def test_against_ks_1samp(self, alternative, a):
        rng = np.random.default_rng(65723433)
        x = stats.skewnorm.rvs(a=a, size=30, random_state=rng)
        expected = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative)

        def statistic1d(x):
            return stats.ks_1samp(x, stats.norm.cdf, mode='asymp', alternative=alternative).statistic
        norm_rvs = self.rvs(stats.norm.rvs, rng)
        res = monte_carlo_test(x, norm_rvs, statistic1d, n_resamples=1000, vectorized=False, alternative=alternative)
        assert_allclose(res.statistic, expected.statistic)
        if alternative == 'greater':
            assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)
        elif alternative == 'less':
            assert_allclose(1 - res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('hypotest', (stats.skewtest, stats.kurtosistest))
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    @pytest.mark.parametrize('a', np.linspace(-2, 2, 5))
    def test_against_normality_tests(self, hypotest, alternative, a):
        rng = np.random.default_rng(85723405)
        x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)
        expected = hypotest(x, alternative=alternative)

        def statistic(x, axis):
            return hypotest(x, axis=axis).statistic
        norm_rvs = self.rvs(stats.norm.rvs, rng)
        res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True, alternative=alternative)
        assert_allclose(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('a', np.arange(-2, 3))
    def test_against_normaltest(self, a):
        rng = np.random.default_rng(12340513)
        x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)
        expected = stats.normaltest(x)

        def statistic(x, axis):
            return stats.normaltest(x, axis=axis).statistic
        norm_rvs = self.rvs(stats.norm.rvs, rng)
        res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True, alternative='greater')
        assert_allclose(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('a', np.linspace(-0.5, 0.5, 5))
    def test_against_cramervonmises(self, a):
        rng = np.random.default_rng(234874135)
        x = stats.skewnorm.rvs(a=a, size=30, random_state=rng)
        expected = stats.cramervonmises(x, stats.norm.cdf)

        def statistic1d(x):
            return stats.cramervonmises(x, stats.norm.cdf).statistic
        norm_rvs = self.rvs(stats.norm.rvs, rng)
        res = monte_carlo_test(x, norm_rvs, statistic1d, n_resamples=1000, vectorized=False, alternative='greater')
        assert_allclose(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('dist_name', ('norm', 'logistic'))
    @pytest.mark.parametrize('i', range(5))
    def test_against_anderson(self, dist_name, i):

        def fun(a):
            rng = np.random.default_rng(394295467)
            x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
            expected = stats.anderson(x, dist_name)
            return expected.statistic - expected.critical_values[i]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sol = root(fun, x0=0)
        assert sol.success
        a = sol.x[0]
        rng = np.random.default_rng(394295467)
        x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
        expected = stats.anderson(x, dist_name)
        expected_stat = expected.statistic
        expected_p = expected.significance_level[i] / 100

        def statistic1d(x):
            return stats.anderson(x, dist_name).statistic
        dist_rvs = self.rvs(getattr(stats, dist_name).rvs, rng)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = monte_carlo_test(x, dist_rvs, statistic1d, n_resamples=1000, vectorized=False, alternative='greater')
        assert_allclose(res.statistic, expected_stat)
        assert_allclose(res.pvalue, expected_p, atol=2 * self.atol)

    def test_p_never_zero(self):
        rng = np.random.default_rng(2190176673029737545)
        x = np.zeros(100)
        res = monte_carlo_test(x, rng.random, np.mean, vectorized=True, alternative='less')
        assert res.pvalue == 0.0001

    def test_against_ttest_ind(self):
        rng = np.random.default_rng(219017667302737545)
        data = (rng.random(size=(2, 5)), rng.random(size=7))
        rvs = (rng.normal, rng.normal)

        def statistic(x, y, axis):
            return stats.ttest_ind(x, y, axis).statistic
        res = stats.monte_carlo_test(data, rvs, statistic, axis=-1)
        ref = stats.ttest_ind(data[0], [data[1]], axis=-1)
        assert_allclose(res.statistic, ref.statistic)
        assert_allclose(res.pvalue, ref.pvalue, rtol=0.02)

    def test_against_f_oneway(self):
        rng = np.random.default_rng(219017667302737545)
        data = (rng.random(size=(2, 100)), rng.random(size=(2, 101)), rng.random(size=(2, 102)), rng.random(size=(2, 103)))
        rvs = (rng.normal, rng.normal, rng.normal, rng.normal)

        def statistic(*args, axis):
            return stats.f_oneway(*args, axis=axis).statistic
        res = stats.monte_carlo_test(data, rvs, statistic, axis=-1, alternative='greater')
        ref = stats.f_oneway(*data, axis=-1)
        assert_allclose(res.statistic, ref.statistic)
        assert_allclose(res.pvalue, ref.pvalue, atol=0.01)