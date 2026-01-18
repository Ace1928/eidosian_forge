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
class TestStaticFactor(CheckDynamicFactor):
    """
    Test for a static factor model (i.e. factors are not autocorrelated).
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sfm.copy()
        true['predict'] = output_results.iloc[1:][['predict_sfm_1', 'predict_sfm_2', 'predict_sfm_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sfm_1', 'dyn_predict_sfm_2', 'dyn_predict_sfm_3']]
        super().setup_class(true, k_factors=1, factor_order=0)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-05)

    def test_bic(self):
        pass