import numpy as np
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._diagnostic_other import CMTNewey, CMTTauchen
import statsmodels.stats._diagnostic_other as diao
class CheckCMT:

    def test_score(self):
        expected = self.results_score
        for msg, actual in self.res_score():
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13, err_msg=msg)

    def test_scorehc0(self):
        expected = self.results_hc0
        for msg, actual in self.res_hc0():
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13, err_msg=msg)

    def test_scoreopg(self):
        expected = self.results_opg
        for msg, actual in self.res_opg():
            assert_allclose(actual, expected[:np.size(actual)], rtol=1e-13, err_msg=msg)