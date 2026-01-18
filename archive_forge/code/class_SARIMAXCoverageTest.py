import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
class SARIMAXCoverageTest:

    @classmethod
    def setup_class(cls, i, decimal=4, endog=None, *args, **kwargs):
        if endog is None:
            endog = results_sarimax.wpi1_data
        cls.true_loglike = coverage_results.loc[i]['llf']
        cls.true_params = np.array([float(x) for x in coverage_results.loc[i]['parameters'].split(',')])
        cls.true_params[-1] = cls.true_params[-1] ** 2
        cls.decimal = decimal
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)
        cls.model = sarimax.SARIMAX(endog, *args, **kwargs)

    def test_loglike(self):
        self.result = self.model.filter(self.true_params)
        assert_allclose(self.result.llf, self.true_loglike, atol=0.7 * 10 ** (-self.decimal))

    def test_start_params(self):
        stat = self.model.enforce_stationarity
        inv = self.model.enforce_invertibility
        self.model.enforce_stationarity = False
        self.model.enforce_invertibility = False
        self.model.start_params
        self.model.enforce_stationarity = stat
        self.model.enforce_invertibility = inv

    def test_transform_untransform(self):
        model = self.model
        stat, inv = (model.enforce_stationarity, model.enforce_invertibility)
        true_constrained = self.true_params
        model.update(self.true_params)
        par = model.polynomial_ar
        psar = model.polynomial_seasonal_ar
        contracted_psar = psar[psar.nonzero()]
        model.enforce_stationarity = (model.k_ar == 0 or tools.is_invertible(np.r_[1, -par[1:]])) and (len(contracted_psar) <= 1 or tools.is_invertible(np.r_[1, -contracted_psar[1:]]))
        pma = model.polynomial_ma
        psma = model.polynomial_seasonal_ma
        contracted_psma = psma[psma.nonzero()]
        model.enforce_invertibility = (model.k_ma == 0 or tools.is_invertible(np.r_[1, pma[1:]])) and (len(contracted_psma) <= 1 or tools.is_invertible(np.r_[1, contracted_psma[1:]]))
        unconstrained = model.untransform_params(true_constrained)
        constrained = model.transform_params(unconstrained)
        assert_almost_equal(constrained, true_constrained, 4)
        model.enforce_stationarity = stat
        model.enforce_invertibility = inv

    def test_results(self):
        self.result = self.model.filter(self.true_params)
        self.result.summary()
        self.result.cov_params_default
        self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg
        self.result.cov_params_robust_oim
        self.result.cov_params_robust_approx

    @pytest.mark.matplotlib
    def test_plot_diagnostics(self, close_figures):
        self.result = self.model.filter(self.true_params)
        self.result.plot_diagnostics()

    def test_predict(self):
        result = self.model.filter(self.true_params)
        predict = result.predict()
        assert_equal(predict.shape, (self.model.nobs,))
        predict = result.predict(start=10, end=20)
        assert_equal(predict.shape, (11,))
        predict = result.predict(start=10, end=20, dynamic=10)
        assert_equal(predict.shape, (11,))
        if self.model.k_exog == 0:
            predict = result.predict(start=self.model.nobs, end=self.model.nobs + 10, dynamic=-10)
            assert_equal(predict.shape, (11,))
            predict = result.predict(start=self.model.nobs, end=self.model.nobs + 10, dynamic=-10)
            forecast = result.forecast()
            assert_equal(forecast.shape, (1,))
            forecast = result.forecast(10)
            assert_equal(forecast.shape, (10,))
        else:
            k_exog = self.model.k_exog
            exog = np.r_[[0] * k_exog * 11].reshape(11, k_exog)
            predict = result.predict(start=self.model.nobs, end=self.model.nobs + 10, dynamic=-10, exog=exog)
            assert_equal(predict.shape, (11,))
            predict = result.predict(start=self.model.nobs, end=self.model.nobs + 10, dynamic=-10, exog=exog)
            exog = np.r_[[0] * k_exog].reshape(1, k_exog)
            forecast = result.forecast(exog=exog)
            assert_equal(forecast.shape, (1,))

    def test_init_keys_replicate(self):
        mod1 = self.model
        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog
        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-06 if PLATFORM_WIN else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)