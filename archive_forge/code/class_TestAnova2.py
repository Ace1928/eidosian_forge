from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova2(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([3.067066, 13.27205, 0.1905093, 27.60181])
        Df = np.array([1, 2, 2, 51])
        F_value = np.array([5.667033, 12.26141, 0.1760025, np.nan])
        PrF = np.array([0.02106078, 4.487909e-05, 0.8391231, np.nan])
        results = anova_lm(anova_ii, typ='II')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)