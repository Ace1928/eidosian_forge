from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova3(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F_value = np.array([279.7545, 5.367071, 12.43245, 0.1760025, np.nan])
        PrF = np.array([2.379855e-22, 0.02457384, 3.999431e-05, 0.8391231, np.nan])
        results = anova_lm(anova_iii, typ='III')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, Sum_Sq, 4)
        np.testing.assert_almost_equal(results['F'].values, F_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)