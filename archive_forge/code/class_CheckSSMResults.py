from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
class CheckSSMResults:
    atol = 1e-14
    rtol = 1e-07
    atol_diffuse = 1e-07
    rtol_diffuse = None

    def check_object(self, actual, desired, rtol_diffuse):
        if actual is None or desired is None:
            return
        d = None
        if rtol_diffuse is None:
            rtol_diffuse = self.rtol_diffuse
        if rtol_diffuse is not None:
            d = self.d
            if rtol_diffuse != np.inf:
                assert_allclose(actual.T[:d], desired.T[:d], rtol=rtol_diffuse, atol=self.atol_diffuse)
        assert_allclose(actual.T[d:], desired.T[d:], rtol=self.rtol, atol=self.atol)

    def test_forecasts(self, rtol_diffuse=None):
        actual = self.results_a.forecasts
        desired = self.results_a.forecasts
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error(self, rtol_diffuse=None):
        actual = self.results_a.forecasts_error
        desired = self.results_a.forecasts_error
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error_cov(self, rtol_diffuse=None):
        actual = self.results_a.forecasts_error_cov
        desired = self.results_b.forecasts_error_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_filtered_state(self, rtol_diffuse=1e-05):
        actual = self.results_a.filtered_state
        desired = self.results_b.filtered_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_filtered_state_cov(self, rtol_diffuse=None):
        actual = self.results_a.filtered_state_cov
        desired = self.results_b.filtered_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_state(self, rtol_diffuse=None):
        actual = self.results_a.predicted_state
        desired = self.results_b.predicted_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_state_cov(self, rtol_diffuse=None):
        actual = self.results_a.predicted_state_cov
        desired = self.results_b.predicted_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_kalman_gain(self, rtol_diffuse=None):
        actual = self.results_a.kalman_gain
        desired = self.results_b.kalman_gain
        self.check_object(actual, desired, rtol_diffuse)

    def test_loglike(self, rtol_diffuse=None):
        if np.isscalar(self.results_b.llf_obs):
            actual = np.sum(self.results_a.llf_obs)
            desired = self.results_b.llf_obs
            assert_allclose(actual, desired)
        else:
            actual = self.results_a.llf_obs
            desired = self.results_b.llf_obs
            self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_state
        desired = self.results_b.smoothed_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_cov(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_state_cov
        desired = self.results_b.smoothed_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_autocov(self, rtol_diffuse=None):
        actual = self.results_a.smoothed_state_autocov
        desired = self.results_b.smoothed_state_autocov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_measurement_disturbance(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_measurement_disturbance
        desired = self.results_b.smoothed_measurement_disturbance
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_measurement_disturbance_cov(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_measurement_disturbance_cov
        desired = self.results_b.smoothed_measurement_disturbance_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_disturbance(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_state_disturbance
        desired = self.results_b.smoothed_state_disturbance
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_disturbance_cov(self, rtol_diffuse=1e-05):
        actual = self.results_a.smoothed_state_disturbance_cov
        desired = self.results_b.smoothed_state_disturbance_cov
        self.check_object(actual, desired, rtol_diffuse)

    @pytest.mark.skip('This is not computed in the univariate method or by KFAS.')
    def test_smoothing_error(self, rtol_diffuse=None):
        actual = self.results_a.smoothing_error
        desired = self.results_b.smoothing_error
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_estimator(self, rtol_diffuse=1e-05):
        actual = self.results_a.scaled_smoothed_estimator
        desired = self.results_b.scaled_smoothed_estimator
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_estimator_cov(self, rtol_diffuse=1e-05):
        actual = self.results_a.scaled_smoothed_estimator_cov
        desired = self.results_b.scaled_smoothed_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error_diffuse_cov(self, rtol_diffuse=None):
        actual = self.results_a.forecasts_error_diffuse_cov
        desired = self.results_b.forecasts_error_diffuse_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_diffuse_state_cov(self, rtol_diffuse=None):
        actual = self.results_a.predicted_diffuse_state_cov
        desired = self.results_b.predicted_diffuse_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse_estimator(self, rtol_diffuse=None):
        actual = self.results_a.scaled_smoothed_diffuse_estimator
        desired = self.results_b.scaled_smoothed_diffuse_estimator
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse1_estimator_cov(self, rtol_diffuse=None):
        actual = self.results_a.scaled_smoothed_diffuse1_estimator_cov
        desired = self.results_b.scaled_smoothed_diffuse1_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse2_estimator_cov(self, rtol_diffuse=None):
        actual = self.results_a.scaled_smoothed_diffuse2_estimator_cov
        desired = self.results_b.scaled_smoothed_diffuse2_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_state(self):
        assert_allclose(self.sim_a.simulated_state, self.sim_a.simulated_state)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(self.sim_a.simulated_measurement_disturbance, self.sim_a.simulated_measurement_disturbance)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(self.sim_a.simulated_state_disturbance, self.sim_a.simulated_state_disturbance)