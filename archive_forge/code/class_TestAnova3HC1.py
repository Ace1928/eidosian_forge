from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnova3HC1(TestAnovaLM):

    def test_results(self):
        data = self.data.drop([0, 1, 2])
        anova_iii = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', data).fit()
        Sum_Sq = np.array([151.4065, 2.904723, 13.45718, 0.1905093, 27.60181])
        Df = np.array([1, 1, 2, 2, 51])
        F = np.array([266.9361, 5.12115, 12.3122, 0.1529943, np.nan])
        PrF = np.array([6.54355e-22, 0.02792296, 4.336712e-05, 0.858527, np.nan])
        results = anova_lm(anova_iii, typ='III', robust='hc1')
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['F'].values, F, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, PrF)