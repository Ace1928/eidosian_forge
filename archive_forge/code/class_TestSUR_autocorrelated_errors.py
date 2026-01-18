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
class TestSUR_autocorrelated_errors(CheckDynamicFactor):
    """
    Test for a seemingly unrelated regression model (i.e. no factors) where
    the errors are vector autocorrelated, but innovations are uncorrelated.
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sur_auto.copy()
        true['predict'] = output_results.iloc[1:][['predict_sur_auto_1', 'predict_sur_auto_2']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sur_auto_1', 'dyn_predict_sur_auto_2']]
        exog = np.c_[np.ones((75, 1)), (np.arange(75) + 2)[:, np.newaxis]]
        super().setup_class(true, k_factors=0, factor_order=0, exog=exog, error_order=1, error_var=True, error_cov_type='diagonal', included_vars=['dln_inv', 'dln_inc'])

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-05)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_dynamic_predict(exog=exog)

    def test_mle(self):
        super().test_mle(init_powell=False)