from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnovaLMCompareNoconstant(TestAnovaLM):

    def test_results(self):
        new_model = ols('np.log(Days+1) ~ C(Duration) + C(Weight) - 1', self.data).fit()
        results = anova_lm(new_model, self.kidney_lm)
        Res_Df = np.array([56, 54])
        RSS = np.array([29.62486, 28.9892])
        Df = np.array([0, 2])
        Sum_of_Sq = np.array([np.nan, 0.6356584])
        F = np.array([np.nan, 0.5920404])
        PrF = np.array([np.nan, 0.5567479])
        np.testing.assert_equal(results['df_resid'].values, Res_Df)
        np.testing.assert_almost_equal(results['ssr'].values, RSS, 4)
        np.testing.assert_almost_equal(results['df_diff'].values, Df)
        np.testing.assert_almost_equal(results['ss_diff'].values, Sum_of_Sq)
        np.testing.assert_almost_equal(results['F'].values, F)
        np.testing.assert_almost_equal(results['Pr(>F)'].values, PrF)