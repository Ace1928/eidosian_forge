import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
class TestFedFundsConstL1(MarkovRegression):

    @classmethod
    def setup_class(cls):
        true = {'params': np.r_[0.6378175, 0.1306295, 0.724457, -0.0988764, 0.7631424, 1.061174, 0.6915759 ** 2], 'llf': -264.71069, 'llf_fit': -264.71069, 'llf_fit_em': -264.71153, 'bse_oim': np.r_[0.1202616, 0.0495924, 0.2886657, 0.1183838, 0.0337234, 0.0185031, np.nan]}
        super().setup_class(true, fedfunds[1:], k_regimes=2, exog=fedfunds[:-1])

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:-1], self.true['bse_oim'][:-1], atol=1e-06)