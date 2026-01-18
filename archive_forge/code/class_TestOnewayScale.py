import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
class TestOnewayScale:

    @classmethod
    def setup_class(cls):
        yt0 = np.array([102.0, 320.0, 0.0, 107.0, 198.0, 200.0, 4.0, 20.0, 110.0, 128.0, 7.0, 119.0, 309.0])
        yt1 = np.array([0.0, 1.0, 228.0, 81.0, 87.0, 119.0, 79.0, 181.0, 43.0, 12.0, 90.0, 105.0, 108.0, 119.0, 0.0, 9.0])
        yt2 = np.array([33.0, 294.0, 134.0, 216.0, 83.0, 105.0, 69.0, 20.0, 20.0, 63.0, 98.0, 155.0, 78.0, 75.0])
        y0 = np.array([452.0, 874.0, 554.0, 447.0, 356.0, 754.0, 558.0, 574.0, 664.0, 682.0, 547.0, 435.0, 245.0])
        y1 = np.array([546.0, 547.0, 774.0, 465.0, 459.0, 665.0, 467.0, 365.0, 589.0, 534.0, 456.0, 651.0, 654.0, 665.0, 546.0, 537.0])
        y2 = np.array([785.0, 458.0, 886.0, 536.0, 669.0, 857.0, 821.0, 772.0, 732.0, 689.0, 654.0, 597.0, 830.0, 827.0])
        n_groups = 3
        data = [y0, y1, y2]
        nobs = np.asarray([len(yi) for yi in data])
        nobs_mean = np.mean(nobs)
        means = np.asarray([yi.mean() for yi in data])
        stds = np.asarray([yi.std(ddof=1) for yi in data])
        cls.data = data
        cls.data_transformed = [yt0, yt1, yt2]
        cls.means = means
        cls.nobs = nobs
        cls.stds = stds
        cls.n_groups = n_groups
        cls.nobs_mean = nobs_mean

    def test_means(self):
        statistic = 7.10900606421182
        parameter = [2, 31.4207256105052]
        p_value = 0.00283841965791224
        res = anova_oneway(self.data, use_var='bf')
        assert_allclose(res.pvalue2, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose([res.df_num2, res.df_denom], parameter)

    def test_levene(self):
        data = self.data
        statistic = 1.0866123063642
        p_value = 0.3471072204516
        res0 = smo.test_scale_oneway(data, method='equal', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        statistic = 1.10732113109744
        p_value = 0.340359251994645
        df = [2, 40]
        res0 = smo.test_scale_oneway(data, method='equal', center='trimmed', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, df)
        statistic = 1.07894485177512
        parameter = [2, 40]
        p_value = 0.349641166869223
        res0 = smo.test_scale_oneway(data, method='equal', center='mean', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res0.pvalue, p_value, rtol=1e-13)
        assert_allclose(res0.statistic, statistic, rtol=1e-13)
        assert_allclose(res0.df, parameter)
        statistic = 3.01982414477323
        p_value = 0.220929402900495
        from scipy import stats
        stat, pv = stats.bartlett(*data)
        assert_allclose(pv, p_value, rtol=1e-13)
        assert_allclose(stat, statistic, rtol=1e-13)

    def test_options(self):
        data = self.data
        statistic, p_value = (1.0173464626246675, 0.3763806150460239)
        df = (2.0, 24.40374758005409)
        res = smo.test_scale_oneway(data, method='unequal', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        statistic, p_value = (1.0329722145270606, 0.3622778213868562)
        df = (1.83153791573948, 30.6733640949525)
        p_value2 = 0.3679999679787619
        df2 = (2, 30.6733640949525)
        res = smo.test_scale_oneway(data, method='bf', center='median', transform='abs', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        assert_allclose(res.pvalue2, p_value2, rtol=1e-13)
        assert_allclose(res.df2, df2)
        statistic, p_value = (1.7252431333701745, 0.19112038168209514)
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean', transform='square', trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_equal(res.df, df)
        statistic, p_value = (0.4129696057329463, 0.6644711582864451)
        df = (2.0, 40.0)
        res = smo.test_scale_oneway(data, method='equal', center='mean', transform=lambda x: np.log(x * x), trim_frac_mean=0.2)
        assert_allclose(res.pvalue, p_value, rtol=1e-13)
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.df, df)
        res = smo.test_scale_oneway(data, method='unequal', center=0, transform='identity', trim_frac_mean=0.2)
        res2 = anova_oneway(self.data, use_var='unequal')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)

    def test_equivalence(self):
        data = self.data
        res = smo.equivalence_scale_oneway(data, 0.5, method='unequal', center=0, transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var='unequal')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)
        res = smo.equivalence_scale_oneway(data, 0.5, method='bf', center=0, transform='identity')
        res2 = equivalence_oneway(self.data, 0.5, use_var='bf')
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-13)
        assert_allclose(res.statistic, res2.statistic, rtol=1e-13)
        assert_allclose(res.df, res2.df)