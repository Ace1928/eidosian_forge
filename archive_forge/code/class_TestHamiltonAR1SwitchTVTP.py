import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
class TestHamiltonAR1SwitchTVTP(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[6.564923, 7.846371, -8.064123, -15.37636, 1.02719, -0.71976, np.exp(-0.217003) ** 2, 0.161489, 0.022536], 'llf': -163.914049, 'llf_fit': -161.786477, 'llf_fit_em': -163.914049}
        exog_tvtp = np.c_[np.ones(len(rgnp)), rec]
        super().setup_class(true, rgnp, k_regimes=2, order=1, exog_tvtp=exog_tvtp)

    @pytest.mark.skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(self.result.filtered_marginal_probabilities[:, 0], hamilton_ar1_switch_tvtp_filtered, atol=1e-05)

    def test_smoothed_regimes(self):
        assert_allclose(self.result.smoothed_marginal_probabilities[:, 0], hamilton_ar1_switch_tvtp_smoothed, atol=1e-05)

    def test_expected_durations(self):
        assert_allclose(self.result.expected_durations, expected_durations, rtol=1e-05, atol=1e-07)