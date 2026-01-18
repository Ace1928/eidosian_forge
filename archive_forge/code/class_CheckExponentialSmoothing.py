import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class CheckExponentialSmoothing:

    @classmethod
    def setup_class(cls, name, res):
        cls.name = name
        cls.res = res
        cls.nobs = res.nobs
        cls.nforecast = len(results_predict['%s_mean' % cls.name]) - cls.nobs
        cls.forecast = res.get_forecast(cls.nforecast)

    def test_fitted(self):
        predicted = results_predict['%s_mean' % self.name]
        assert_allclose(self.res.fittedvalues, predicted.iloc[:self.nobs])

    def test_output(self):
        has_llf = ~np.isnan(results_params[self.name]['llf'])
        if has_llf:
            assert_allclose(self.res.mse, results_params[self.name]['mse'])
            actual = -0.5 * self.nobs * np.log(np.sum(self.res.resid ** 2))
            assert_allclose(actual, results_params[self.name]['llf'])
        else:
            assert_allclose(self.res.sse, results_params[self.name]['sse'])

    def test_forecasts(self):
        predicted = results_predict['%s_mean' % self.name]
        assert_allclose(self.forecast.predicted_mean, predicted.iloc[self.nobs:])

    def test_conf_int(self):
        ci_95 = self.forecast.conf_int(alpha=0.05)
        lower = results_predict['%s_lower' % self.name]
        upper = results_predict['%s_upper' % self.name]
        assert_allclose(ci_95['lower y'], lower.iloc[self.nobs:])
        assert_allclose(ci_95['upper y'], upper.iloc[self.nobs:])

    def test_initial_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[0]
        assert_allclose(self.res.initial_state.iloc[0], desired)

    def test_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[1:]
        assert_allclose(self.res.filtered_state[1:].T, desired)

    def test_misc(self):
        mod = self.res.model
        assert_equal(mod.k_params, len(mod.start_params))
        assert_equal(mod.k_params, len(mod.param_names))
        self.res.summary()