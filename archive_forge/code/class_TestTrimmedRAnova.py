import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pytest
from statsmodels.stats.robust_compare import (
import statsmodels.stats.oneway as smo
from statsmodels.tools.testing import Holder
from scipy.stats import trim1
class TestTrimmedRAnova:

    @classmethod
    def setup_class(cls):
        x = [np.array([452.0, 874.0, 554.0, 447.0, 356.0, 754.0, 558.0, 574.0, 664.0, 682.0, 547.0, 435.0, 245.0]), np.array([546.0, 547.0, 774.0, 465.0, 459.0, 665.0, 467.0, 365.0, 589.0, 534.0, 456.0, 651.0, 654.0, 665.0, 546.0, 537.0]), np.array([785.0, 458.0, 886.0, 536.0, 669.0, 857.0, 821.0, 772.0, 732.0, 689.0, 654.0, 597.0, 830.0, 827.0])]
        cls.x = x
        cls.get_results()

    @classmethod
    def get_results(cls):
        cls.res_m = [549.3846153846154, 557.5, 722.3571428571429]
        cls.res_oneway = Holder(test=8.81531710400927, df1=2, df2=19.8903710685394, p_value=0.00181464966984701, effsize=0.647137153056774)
        cls.res_2s = Holder(test=0.161970203096559, conf_int=np.array([-116.437383793431, 99.9568643129114]), p_value=0.873436269777141, df=15.3931262881751, diff=-8.24025974025983, effsize=0.0573842557922749)
        cls.res_bfm = Holder(statistic=7.10900606421182, parameter=np.array([2, 31.4207256105052]), p_value=0.00283841965791224, alpha=0.05, method='Brown-Forsythe Test')
        cls.res_wa = Holder(statistic=8.02355212103924, parameter=np.array([2, 24.272320628139]), p_value=0.00211423625518082, method='One-way analysis of means (not assuming equal variances)')
        cls.res_fa = Holder(statistic=7.47403193349076, parameter=np.array([2, 40]), p_value=0.00174643304119871, method='One-way analysis of means')

    def test_oneway(self):
        r1 = self.res_oneway
        r2s = self.res_2s
        res_bfm = self.res_bfm
        res_wa = self.res_wa
        res_fa = self.res_fa
        m = [x_i.mean() for x_i in self.x]
        assert_allclose(m, self.res_m, rtol=1e-13)
        resg = smo.anova_oneway(self.x, use_var='unequal', trim_frac=1 / 13)
        assert_allclose(resg.pvalue, r1.p_value, rtol=1e-13)
        assert_allclose(resg.df, [r1.df1, r1.df2], rtol=1e-13)
        resg = smo.anova_oneway(self.x[:2], use_var='unequal', trim_frac=1 / 13)
        assert_allclose(resg.pvalue, r2s.p_value, rtol=1e-13)
        assert_allclose(resg.df, [1, r2s.df], rtol=1e-13)
        res = smo.anova_oneway(self.x, use_var='bf')
        assert_allclose(res[0], res_bfm.statistic, rtol=1e-13)
        assert_allclose(res.pvalue2, res_bfm.p_value, rtol=1e-13)
        assert_allclose(res.df2, res_bfm.parameter, rtol=1e-13)
        res = smo.anova_oneway(self.x, use_var='unequal')
        assert_allclose(res.statistic, res_wa.statistic, rtol=1e-13)
        assert_allclose(res.pvalue, res_wa.p_value, rtol=1e-13)
        assert_allclose(res.df, res_wa.parameter, rtol=1e-13)
        res = smo.anova_oneway(self.x, use_var='equal')
        assert_allclose(res.statistic, res_fa.statistic, rtol=1e-13)
        assert_allclose(res.pvalue, res_fa.p_value, rtol=1e-13)
        assert_allclose(res.df, res_fa.parameter, rtol=1e-13)