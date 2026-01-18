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
class CheckDynamicFactor:

    @classmethod
    def setup_class(cls, true, k_factors, factor_order, cov_type='approx', included_vars=['dln_inv', 'dln_inc', 'dln_consump'], demean=False, filter=True, **kwargs):
        cls.true = true
        dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()
        endog = dta.loc['1960-04-01':'1978-10-01', included_vars]
        if demean:
            endog -= dta.iloc[1:][included_vars].mean()
        cls.model = dynamic_factor.DynamicFactor(endog, k_factors=k_factors, factor_order=factor_order, **kwargs)
        if filter:
            cls.results = cls.model.smooth(true['params'], cov_type=cov_type)

    def test_params(self):
        self.model.filter(self.model.start_params)
        assert_equal(len(self.model.start_params), len(self.model.param_names))
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        assert_allclose(actual, self.model.start_params)
        self.model.enforce_stationarity = False
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        self.model.enforce_stationarity = True
        assert_allclose(actual, self.model.start_params)

    def test_results(self, close_figures):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.results.summary()
        if self.model.factor_order > 0:
            model = self.model
            k_factors = model.k_factors
            pft_params = self.results.params[model._params_factor_transition]
            coefficients = np.array(pft_params).reshape(k_factors, k_factors * model.factor_order)
            coefficient_matrices = np.array([coefficients[:self.model.k_factors, i * self.model.k_factors:(i + 1) * self.model.k_factors] for i in range(self.model.factor_order)])
            assert_equal(self.results.coefficient_matrices_var, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)

    @pytest.mark.matplotlib
    def test_plot_coefficients_of_determination(self, close_figures):
        self.results.plot_coefficients_of_determination()

    def test_no_enforce(self):
        return
        params = self.model.untransform_params(self.true['params'])
        params[self.model._params_transition] = self.true['params'][self.model._params_transition]
        self.model.enforce_stationarity = False
        results = self.model.filter(params, transformed=False)
        self.model.enforce_stationarity = True
        assert_allclose(results.llf, self.results.llf, rtol=1e-05)

    def test_mle(self, init_powell=True):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            start_params = self.model.start_params
            if init_powell:
                results = self.model.fit(method='powell', maxiter=100, disp=False)
                start_params = results.params
            results = self.model.fit(start_params, maxiter=1000, disp=False)
            results = self.model.fit(results.params, method='nm', maxiter=1000, disp=False)
            if not results.llf > self.results.llf:
                assert_allclose(results.llf, self.results.llf, rtol=1e-05)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-06)

    def test_aic(self):
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, **kwargs):
        self.results.predict(end='1982-10-01', **kwargs)
        assert_allclose(self.results.predict(end='1982-10-01', **kwargs), self.true['predict'], atol=1e-06)

    def test_dynamic_predict(self, **kwargs):
        assert_allclose(self.results.predict(end='1982-10-01', dynamic='1961-01-01', **kwargs), self.true['dynamic_predict'], atol=1e-06)