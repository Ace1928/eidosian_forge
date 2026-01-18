from io import StringIO
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from pandas import read_csv
class TestAnovaLM:

    @classmethod
    def setup_class(cls):
        cls.data = kidney_table
        cls.kidney_lm = ols('np.log(Days+1) ~ C(Duration) * C(Weight)', data=cls.data).fit()

    def test_results(self):
        Df = np.array([1, 2, 2, 54])
        sum_sq = np.array([2.339693, 16.97129, 0.6356584, 28.9892])
        mean_sq = np.array([2.339693, 8.485645, 0.3178292, 0.536837])
        f_value = np.array([4.358293, 15.80674, 0.5920404, np.nan])
        pr_f = np.array([0.0415617, 3.944502e-06, 0.5567479, np.nan])
        results = anova_lm(self.kidney_lm)
        np.testing.assert_equal(results['df'].values, Df)
        np.testing.assert_almost_equal(results['sum_sq'].values, sum_sq, 4)
        np.testing.assert_almost_equal(results['F'].values, f_value, 4)
        np.testing.assert_almost_equal(results['PR(>F)'].values, pr_f)