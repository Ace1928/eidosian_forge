from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova2Noconstant(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_ii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum) - 1', data).fit()
        Sum_Sq = np.array([154.7131692, 13.27205, 0.1905093, 27.60181])
        Df = np.array([2, 2, 2, 51])
        F_value = np.array([142.9321191, 12.26141, 0.1760025, np.nan])
        PrF = np.array([1.238624e-21, 4.487909e-05, 0.8391231, np.nan])
        results = anova_lm(anova_ii, typ='II')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)