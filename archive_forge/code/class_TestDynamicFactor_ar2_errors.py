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
class TestDynamicFactor_ar2_errors(CheckDynamicFactor):
    """
    Test for a dynamic factor model where errors are as general as possible,
    meaning:

    - Errors are vector autocorrelated, VAR(1)
    - Innovations are correlated
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_ar2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_ar2_1', 'predict_dfm_ar2_2', 'predict_dfm_ar2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_ar2_1', 'dyn_predict_dfm_ar2_2', 'dyn_predict_dfm_ar2_3']]
        super().setup_class(true, k_factors=1, factor_order=1, error_order=2)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-05)

    def test_mle(self):
        with warnings.catch_warnings(record=True):
            mod = self.model
            res1 = mod.fit(maxiter=100, optim_score='approx', disp=False)
            res = mod.fit(res1.params, method='nm', maxiter=10000, optim_score='approx', disp=False)
            assert_allclose(res.llf, self.results.llf, atol=0.01, rtol=0.0001)