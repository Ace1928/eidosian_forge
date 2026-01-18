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
class CheckVARMAX:
    """
    Test Vector Autoregression against Stata's `dfactor` code (Stata's
    `var` function uses OLS and not state space / MLE, so we cannot get
    equivalent log-likelihoods)
    """

    def test_mle(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            results = self.model.fit(maxiter=100, disp=False)
            self.model.enforce_stationarity = False
            self.model.enforce_invertibility = False
            results = self.model.fit(results.params, method='nm', maxiter=1000, disp=False)
            self.model.enforce_stationarity = True
            self.model.enforce_invertibility = True
            assert_allclose(results.llf, self.results.llf, rtol=1e-05)

    @pytest.mark.smoke
    def test_params(self):
        model = self.model
        model.filter(model.start_params)
        assert len(model.start_params) == len(model.param_names)
        actual = model.transform_params(model.untransform_params(model.start_params))
        assert_allclose(actual, model.start_params)
        model.enforce_stationarity = False
        model.enforce_invertibility = False
        actual = model.transform_params(model.untransform_params(model.start_params))
        model.enforce_stationarity = True
        model.enforce_invertibility = True
        assert_allclose(actual, model.start_params)

    @pytest.mark.smoke
    def test_results(self):
        self.results.summary()
        model = self.model
        if model.k_ar > 0:
            params_ar = np.array(self.results.params[model._params_ar])
            coefficients = params_ar.reshape(model.k_endog, model.k_endog * model.k_ar)
            coefficient_matrices = np.array([coefficients[:model.k_endog, i * model.k_endog:(i + 1) * model.k_endog] for i in range(model.k_ar)])
            assert_equal(self.results.coefficient_matrices_var, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)
        if model.k_ma > 0:
            params_ma = np.array(self.results.params[model._params_ma])
            coefficients = params_ma.reshape(model.k_endog, model.k_endog * model.k_ma)
            coefficient_matrices = np.array([coefficients[:model.k_endog, i * model.k_endog:(i + 1) * model.k_endog] for i in range(model.k_ma)])
            assert_equal(self.results.coefficient_matrices_vma, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_vma, None)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-06)

    def test_aic(self):
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, end, atol=1e-06, **kwargs):
        assert_allclose(self.results.predict(end=end, **kwargs), self.true['predict'], atol=atol)

    def test_dynamic_predict(self, end, dynamic, atol=1e-06, **kwargs):
        assert_allclose(self.results.predict(end=end, dynamic=dynamic, **kwargs), self.true['dynamic_predict'], atol=atol)

    def test_standardized_forecasts_error(self):
        cython_sfe = self.results.standardized_forecasts_error
        self.results._standardized_forecasts_error = None
        python_sfe = self.results.standardized_forecasts_error
        assert_allclose(cython_sfe, python_sfe)