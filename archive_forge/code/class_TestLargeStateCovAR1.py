import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
class TestLargeStateCovAR1:

    @classmethod
    def setup_class(cls):
        pytest.skip('TODO: This test is skipped since an exception is currently raised if k_posdef > k_states. However, this test could be used if models of those types were allowed')
        endog = [0.2, -1.5, -0.3, -0.1, 1.5, 0.2, -0.3, 0.2, 0.5, 0.8]
        params = [0.5, 1]
        mod_desired = sarimax.SARIMAX(endog)
        cls.res_desired = mod_desired.smooth(params)
        mod = LargeStateCovAR1(endog)
        cls.res = mod.smooth(params)

    def test_dimensions(self):
        assert_equal(self.res.filter_results.k_states, 1)
        assert_equal(self.res.filter_results.k_posdef, 2)
        assert_equal(self.res.smoothed_state_disturbance.shape, (2, 10))
        assert_equal(self.res_desired.filter_results.k_states, 1)
        assert_equal(self.res_desired.filter_results.k_posdef, 1)
        assert_equal(self.res_desired.smoothed_state_disturbance.shape, (1, 10))

    def test_loglike(self):
        assert_allclose(self.res.llf_obs, self.res_desired.llf_obs)

    def test_scaled_smoothed_estimator(self):
        assert_allclose(self.res.scaled_smoothed_estimator[0], self.res_desired.scaled_smoothed_estimator[0])

    def test_scaled_smoothed_estimator_cov(self):
        assert_allclose(self.res.scaled_smoothed_estimator_cov[0], self.res_desired.scaled_smoothed_estimator_cov[0])

    def test_forecasts(self):
        assert_allclose(self.res.forecasts, self.res_desired.forecasts)

    def test_forecasts_error(self):
        assert_allclose(self.res.forecasts_error, self.res_desired.forecasts_error)

    def test_forecasts_error_cov(self):
        assert_allclose(self.res.forecasts_error_cov, self.res_desired.forecasts_error_cov)

    def test_predicted_states(self):
        assert_allclose(self.res.predicted_state[0], self.res_desired.predicted_state[0])

    def test_predicted_states_cov(self):
        assert_allclose(self.res.predicted_state_cov[0, 0], self.res_desired.predicted_state_cov[0, 0])

    def test_smoothed_states(self):
        assert_allclose(self.res.smoothed_state[0], self.res_desired.smoothed_state[0])

    def test_smoothed_states_cov(self):
        assert_allclose(self.res.smoothed_state_cov[0, 0], self.res_desired.smoothed_state_cov[0, 0])

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.res.smoothed_state_disturbance[0], self.res_desired.smoothed_state_disturbance[0])
        assert_allclose(self.res.smoothed_state_disturbance[1], 0)

    def test_smoothed_state_disturbance_cov(self):
        assert_allclose(self.res.smoothed_state_disturbance_cov[0, 0], self.res_desired.smoothed_state_disturbance_cov[0, 0])
        assert_allclose(self.res.smoothed_state_disturbance[1, 1], 0)

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(self.res.smoothed_measurement_disturbance, self.res_desired.smoothed_measurement_disturbance)

    def test_smoothed_measurement_disturbance_cov(self):
        assert_allclose(self.res.smoothed_measurement_disturbance_cov, self.res_desired.smoothed_measurement_disturbance_cov)