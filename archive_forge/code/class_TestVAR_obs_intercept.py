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
class TestVAR_obs_intercept(CheckLutkepohl):

    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_obs_intercept.copy()
        true['predict'] = var_results.iloc[1:][['predict_int1', 'predict_int2', 'predict_int3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_int1', 'dyn_predict_int2', 'dyn_predict_int3']]
        super().setup_class(true, order=(1, 0), trend='n', error_cov_type='diagonal', obs_intercept=true['obs_intercept'])

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=0.0001)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=0.01)

    def test_aic(self):
        pass

    def test_bic(self):
        pass