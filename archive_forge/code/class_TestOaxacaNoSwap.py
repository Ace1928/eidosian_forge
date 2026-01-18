import numpy as np
from statsmodels.datasets.ccard.data import load_pandas
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.tools.tools import add_constant
class TestOaxacaNoSwap:

    @classmethod
    def setup_class(cls):
        cls.model = OaxacaBlinder(endog, exog, 3, swap=False)

    def test_results(self):
        stata_results = np.array([-158.7504, -83.29674, 162.9978, -238.4515])
        stata_results_pooled = np.array([-158.7504, -130.8095, -27.94091])
        endow, coef, inter, gap = self.model.three_fold().params
        unexp, exp, gap = self.model.two_fold().params
        np.testing.assert_almost_equal(gap, stata_results[0], 3)
        np.testing.assert_almost_equal(endow, stata_results[1], 3)
        np.testing.assert_almost_equal(coef, stata_results[2], 3)
        np.testing.assert_almost_equal(inter, stata_results[3], 3)
        np.testing.assert_almost_equal(gap, stata_results_pooled[0], 3)
        np.testing.assert_almost_equal(exp, stata_results_pooled[1], 3)
        np.testing.assert_almost_equal(unexp, stata_results_pooled[2], 3)