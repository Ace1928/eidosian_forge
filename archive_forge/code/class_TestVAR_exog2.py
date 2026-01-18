import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
class TestVAR_exog2(CheckLutkepohl):

    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog2.copy()
        true['predict'] = var_results.iloc[1:76][['predict_exog2_1', 'predict_exog2_2', 'predict_exog2_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.iloc[76:][['fcast_exog2_dln_inv', 'fcast_exog2_dln_inc', 'fcast_exog2_dln_consump']]
        exog = np.c_[np.ones((75, 1)), (np.arange(75) + 2)[:, np.newaxis]]
        super().setup_class(true, order=(1, 0), trend='n', error_cov_type='unstructured', exog=exog, initialization='approximate_diffuse', loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse_approx(self):
        pass

    def test_bse_oim(self):
        pass

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=0.001)

    def test_dynamic_predict(self):
        pass

    def test_forecast(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-06)