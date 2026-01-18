import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
class TestDynamicFactor_exog1(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 1 exogenous regressor: a constant
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_exog1.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_exog1_1', 'predict_dfm_exog1_2', 'predict_dfm_exog1_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_exog1_1', 'dyn_predict_dfm_exog1_2', 'dyn_predict_dfm_exog1_3']]
        exog = np.ones((75, 1))
        super().setup_class(true, k_factors=1, factor_order=1, exog=exog)

    def test_predict(self):
        exog = np.ones((16, 1))
        super().test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.ones((16, 1))
        super().test_dynamic_predict(exog=exog)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=1e-05)