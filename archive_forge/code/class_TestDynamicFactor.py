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
class TestDynamicFactor(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 1 AR(2) factor
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_1', 'predict_dfm_2', 'predict_dfm_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_1', 'dyn_predict_dfm_2', 'dyn_predict_dfm_3']]
        super().setup_class(true, k_factors=1, factor_order=2)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse, self.true['bse_oim'], atol=1e-05)