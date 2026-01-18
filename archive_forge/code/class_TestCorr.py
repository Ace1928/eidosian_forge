import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
class TestCorr:

    def test_pearsonr(self):
        x = ma.arange(10)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_almost_equal(mstats.pearsonr(x, x)[0], 1.0)
            assert_almost_equal(mstats.pearsonr(x, x[::-1])[0], -1.0)
            x = ma.array(x, mask=True)
            pr = mstats.pearsonr(x, x)
            assert_(pr[0] is masked)
            assert_(pr[1] is masked)
        x1 = ma.array([-1.0, 0.0, 1.0])
        y1 = ma.array([0, 0, 3])
        r, p = mstats.pearsonr(x1, y1)
        assert_almost_equal(r, np.sqrt(3) / 2)
        assert_almost_equal(p, 1.0 / 3)
        mask = [False, False, False, True]
        x2 = ma.array([-1.0, 0.0, 1.0, 99.0], mask=mask)
        y2 = ma.array([0, 0, 3, -1], mask=mask)
        r, p = mstats.pearsonr(x2, y2)
        assert_almost_equal(r, np.sqrt(3) / 2)
        assert_almost_equal(p, 1.0 / 3)

    def test_pearsonr_misaligned_mask(self):
        mx = np.ma.masked_array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 0, 0, 0])
        my = np.ma.masked_array([9, 8, 7, 6, 5, 9], mask=[0, 0, 1, 0, 0, 0])
        x = np.array([1, 4, 5, 6])
        y = np.array([9, 6, 5, 9])
        mr, mp = mstats.pearsonr(mx, my)
        r, p = stats.pearsonr(x, y)
        assert_equal(mr, r)
        assert_equal(mp, p)

    def test_spearmanr(self):
        x, y = ([5.05, 6.75, 3.21, 2.66], [1.65, 2.64, 2.64, 6.95])
        assert_almost_equal(mstats.spearmanr(x, y)[0], -0.6324555)
        x, y = ([5.05, 6.75, 3.21, 2.66, np.nan], [1.65, 2.64, 2.64, 6.95, np.nan])
        x, y = (ma.fix_invalid(x), ma.fix_invalid(y))
        assert_almost_equal(mstats.spearmanr(x, y)[0], -0.6324555)
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1, 1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6, 0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]
        assert_almost_equal(mstats.spearmanr(x, y)[0], 0.6887299)
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1, 1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7, np.nan]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6, 0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4, np.nan]
        x, y = (ma.fix_invalid(x), ma.fix_invalid(y))
        assert_almost_equal(mstats.spearmanr(x, y)[0], 0.6887299)
        x = list(range(2000))
        y = list(range(2000))
        y[0], y[9] = (y[9], y[0])
        y[10], y[434] = (y[434], y[10])
        y[435], y[1509] = (y[1509], y[435])
        assert_almost_equal(mstats.spearmanr(x, y)[0], 0.998)
        res = mstats.spearmanr(x, y)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes, ma=True)

    def test_spearmanr_alternative(self):
        x = [2.0, 47.4, 42.0, 10.8, 60.1, 1.7, 64.0, 63.1, 1.0, 1.4, 7.9, 0.3, 3.9, 0.3, 6.7]
        y = [22.6, 8.3, 44.4, 11.9, 24.6, 0.6, 5.7, 41.6, 0.0, 0.6, 6.7, 3.8, 1.0, 1.2, 1.4]
        r_exp = 0.6887298747763864
        r, p = mstats.spearmanr(x, y)
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.004519192910756)
        r, p = mstats.spearmanr(x, y, alternative='greater')
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.002259596455378)
        r, p = mstats.spearmanr(x, y, alternative='less')
        assert_allclose(r, r_exp)
        assert_allclose(p, 0.9977404035446)
        n = 100
        x = np.linspace(0, 5, n)
        y = 0.1 * x + np.random.rand(n)
        stat1, p1 = mstats.spearmanr(x, y)
        stat2, p2 = mstats.spearmanr(x, y, alternative='greater')
        assert_allclose(p2, p1 / 2)
        stat3, p3 = mstats.spearmanr(x, y, alternative='less')
        assert_allclose(p3, 1 - p1 / 2)
        assert stat1 == stat2 == stat3
        with pytest.raises(ValueError, match="alternative must be 'less'..."):
            mstats.spearmanr(x, y, alternative='ekki-ekki')

    @pytest.mark.skipif(platform.machine() == 'ppc64le', reason='fails/crashes on ppc64le')
    def test_kendalltau(self):
        x = ma.array(np.array([9, 2, 5, 6]))
        y = ma.array(np.array([4, 7, 9, 11]))
        expected = [0.0, 1.0]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        x = ma.array(np.arange(10))
        y = ma.array(np.arange(10))
        expected = [1.0, 5.511463844797e-07]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        assert_raises(ValueError, mstats.kendalltau, x, y, method='banana')
        b = y[1]
        y[1] = y[2]
        y[2] = b
        expected = [0.9555555555555556, 5.511463844797e-06]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        b = y[5]
        y[5] = y[6]
        y[6] = b
        expected = [0.9111111111111111, 2.97619047619e-05]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        x = ma.array(np.arange(10))
        y = ma.array(np.arange(10)[::-1])
        expected = [-1.0, 5.511463844797e-07]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        b = y[1]
        y[1] = y[2]
        y[2] = b
        expected = [-0.9555555555555556, 5.511463844797e-06]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        b = y[5]
        y[5] = y[6]
        y[6] = b
        expected = [-0.9111111111111111, 2.97619047619e-05]
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), expected)
        x = ma.fix_invalid([5.05, 6.75, 3.21, 2.66, np.nan])
        y = ma.fix_invalid([1.65, 26.5, -5.93, 7.96, np.nan])
        z = ma.fix_invalid([1.65, 2.64, 2.64, 6.95, np.nan])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y)), [+0.3333333, 0.75])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, y, method='asymptotic')), [+0.3333333, 0.4969059])
        assert_almost_equal(np.asarray(mstats.kendalltau(x, z)), [-0.5477226, 0.2785987])
        x = ma.fix_invalid([0, 0, 0, 0, 20, 20, 0, 60, 0, 20, 10, 10, 0, 40, 0, 20, 0, 0, 0, 0, 0, np.nan])
        y = ma.fix_invalid([0, 80, 80, 80, 10, 33, 60, 0, 67, 27, 25, 80, 80, 80, 80, 80, 80, 0, 10, 45, np.nan, 0])
        result = mstats.kendalltau(x, y)
        assert_almost_equal(np.asarray(result), [-0.1585188, 0.4128009])
        attributes = ('correlation', 'pvalue')
        check_named_results(result, attributes, ma=True)

    @pytest.mark.skipif(platform.machine() == 'ppc64le', reason='fails/crashes on ppc64le')
    @pytest.mark.slow
    def test_kendalltau_large(self):
        x = np.arange(2000, dtype=float)
        x = ma.masked_greater(x, 1995)
        y = np.arange(2000, dtype=float)
        y = np.concatenate((y[1000:], y[:1000]))
        assert_(np.isfinite(mstats.kendalltau(x, y)[1]))

    def test_kendalltau_seasonal(self):
        x = [[nan, nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1], [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3], [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan], [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
        x = ma.fix_invalid(x).T
        output = mstats.kendalltau_seasonal(x)
        assert_almost_equal(output['global p-value (indep)'], 0.008, 3)
        assert_almost_equal(output['seasonal p-value'].round(2), [0.18, 0.53, 0.2, 0.04])

    @pytest.mark.parametrize('method', ('exact', 'asymptotic'))
    @pytest.mark.parametrize('alternative', ('two-sided', 'greater', 'less'))
    def test_kendalltau_mstats_vs_stats(self, method, alternative):
        np.random.seed(0)
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        mask = np.random.rand(n) > 0.5
        x_masked = ma.array(x, mask=mask)
        y_masked = ma.array(y, mask=mask)
        res_masked = mstats.kendalltau(x_masked, y_masked, method=method, alternative=alternative)
        x_compressed = x_masked.compressed()
        y_compressed = y_masked.compressed()
        res_compressed = stats.kendalltau(x_compressed, y_compressed, method=method, alternative=alternative)
        x[mask] = np.nan
        y[mask] = np.nan
        res_nan = stats.kendalltau(x, y, method=method, nan_policy='omit', alternative=alternative)
        assert_allclose(res_masked, res_compressed)
        assert_allclose(res_nan, res_compressed)

    def test_kendall_p_exact_medium(self):
        expectations = {(100, 2393): 0.6282261528795604, (101, 2436): 0.604395257735136, (170, 0): 2.755801935583541e-307, (171, 0): 0.0, (171, 1): 2.755801935583541e-307, (172, 1): 0.0, (200, 9797): 0.7475398374592968, (201, 9656): 0.40959218958120364}
        for nc, expected in expectations.items():
            res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
            assert_almost_equal(res, expected)

    @pytest.mark.xslow
    def test_kendall_p_exact_large(self):
        expectations = {(400, 38965): 0.48444283672113314, (401, 39516): 0.6636315982347484, (800, 156772): 0.4226544848312093, (801, 157849): 0.5343755341219442, (1600, 637472): 0.8420072740032354, (1601, 630304): 0.34465255088058594}
        for nc, expected in expectations.items():
            res = _mstats_basic._kendall_p_exact(nc[0], nc[1])
            assert_almost_equal(res, expected)

    def test_pointbiserial(self):
        x = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1]
        y = [14.8, 13.8, 12.4, 10.1, 7.1, 6.1, 5.8, 4.6, 4.3, 3.5, 3.3, 3.2, 3.0, 2.8, 2.8, 2.5, 2.4, 2.3, 2.1, 1.7, 1.7, 1.5, 1.3, 1.3, 1.2, 1.2, 1.1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1, np.nan]
        assert_almost_equal(mstats.pointbiserialr(x, y)[0], 0.36149, 5)
        res = mstats.pointbiserialr(x, y)
        attributes = ('correlation', 'pvalue')
        check_named_results(res, attributes, ma=True)