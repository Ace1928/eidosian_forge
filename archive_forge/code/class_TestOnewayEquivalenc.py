import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
class TestOnewayEquivalenc:

    @classmethod
    def setup_class(cls):
        y0 = [112.488, 103.738, 86.344, 101.708, 95.108, 105.931, 95.815, 91.864, 102.479, 102.644]
        y1 = [100.421, 101.966, 99.636, 105.983, 88.377, 102.618, 105.486, 98.662, 94.137, 98.626, 89.367, 106.204]
        y2 = [84.846, 100.488, 119.763, 103.736, 93.141, 108.254, 99.51, 89.005, 108.2, 82.209, 100.104, 103.706, 107.067]
        y3 = [100.825, 100.255, 103.363, 93.23, 95.325, 100.288, 94.75, 107.129, 98.246, 96.365, 99.74, 106.049, 92.691, 93.111, 98.243]
        n_groups = 4
        arrs_w = [np.asarray(yi) for yi in [y0, y1, y2, y3]]
        nobs = np.asarray([len(yi) for yi in arrs_w])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in arrs_w])
        stds = np.asarray([yi.std(ddof=1) for yi in arrs_w])
        cls.data = arrs_w
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_equivalence_equal(self):
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        eps = 0.5
        res0 = anova_generic(means, stds ** 2, nobs, use_var='equal')
        f = res0.statistic
        res = equivalence_oneway_generic(f, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])
        assert_allclose(f, 0.0926, atol=0.0006)
        res = equivalence_oneway(self.data, eps, use_var='equal', margin_type='wellek')
        assert_allclose(res.pvalue, 0.0083, atol=0.001)
        assert_equal(res.df, [3, 46])

    def test_equivalence_welch(self):
        means = self.means
        nobs = self.nobs
        stds = self.stds
        n_groups = self.n_groups
        vars_ = stds ** 2
        eps = 0.5
        res0 = anova_generic(means, vars_, nobs, use_var='unequal', welch_correction=False)
        f_stat = res0.statistic
        res = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), eps, res0.df, alpha=0.05, margin_type='wellek')
        assert_allclose(res.pvalue, 0.011, atol=0.001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(f_stat, 0.1102, atol=0.007)
        res = equivalence_oneway(self.data, eps, use_var='unequal', margin_type='wellek')
        assert_allclose(res.pvalue, 0.011, atol=0.0001)
        assert_allclose(res.df, [3.0, 22.6536], atol=0.0006)
        assert_allclose(res.f_stat, 0.1102, atol=0.0001)
        pow_ = _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, res0.df)
        assert_allclose(pow_, 0.1552, atol=0.007)
        pow_ = power_equivalence_oneway(eps, eps, nobs.sum(), n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
        assert_allclose(pow_, 0.05, atol=1e-13)
        nobs_t = nobs.sum()
        es = effectsize_oneway(means, vars_, nobs, use_var='unequal')
        es = np.sqrt(es)
        es_w0 = f2_to_wellek(es ** 2, n_groups)
        es_w = np.sqrt(fstat_to_wellek(f_stat, n_groups, nobs_t / n_groups))
        pow_ = power_equivalence_oneway(es_w, eps, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='wellek')
        assert_allclose(pow_, 0.1552, atol=0.007)
        assert_allclose(es_w0, es_w, atol=0.007)
        margin = wellek_to_f2(eps, n_groups)
        pow_ = power_equivalence_oneway(es ** 2, margin, nobs_t, n_groups=n_groups, df=None, alpha=0.05, margin_type='f2')
        assert_allclose(pow_, 0.1552, atol=0.007)