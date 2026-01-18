import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
class TestHamiltonAR2Switch(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        path = os.path.join(current_path, 'results', 'results_predict_rgnp.csv')
        results = pd.read_csv(path)
        true = {'params': np.r_[0.3812383, 0.3564492, -0.0055216, 1.195482, 0.6677098 ** 2, 0.3710719, 0.4621503, 0.7002937, -0.3206652], 'llf': -179.32354, 'llf_fit': -179.38684, 'llf_fit_em': -184.99606, 'bse_oim': np.r_[0.1424841, 0.0994742, 0.2057086, 0.1225987, np.nan, 0.1754383, 0.1652473, 0.187409, 0.1295937], 'smoothed0': results.iloc[3:]['switchar2_sm1'], 'smoothed1': results.iloc[3:]['switchar2_sm2'], 'predict0': results.iloc[3:]['switchar2_yhat1'], 'predict1': results.iloc[3:]['switchar2_yhat2'], 'predict_predicted': results.iloc[3:]['switchar2_pyhat'], 'predict_filtered': results.iloc[3:]['switchar2_fyhat'], 'predict_smoothed': results.iloc[3:]['switchar2_syhat']}
        super().setup_class(true, rgnp, k_regimes=2, order=2)

    def test_smoothed_marginal_probabilities(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0], self.true['smoothed0'], atol=1e-06)
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 1], self.true['smoothed1'], atol=1e-06)

    def test_predict(self):
        actual = self.model.predict(self.true['params'], probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)
        actual = self.model.predict(self.true['params'], probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)
        actual = self.result.predict(probabilities='smoothed')
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)
        actual = self.result.predict(probabilities=None)
        assert_allclose(actual, self.true['predict_smoothed'], atol=1e-06)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:4], self.true['bse_oim'][:4], atol=1e-07)
        assert_allclose(bse[6:], self.true['bse_oim'][6:], atol=1e-07)