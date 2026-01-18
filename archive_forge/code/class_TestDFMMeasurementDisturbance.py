import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
class TestDFMMeasurementDisturbance(TestDFM):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(smooth_method=SMOOTH_CLASSICAL, which='none', **kwargs)

    def test_smoothed_state_disturbance(self):
        assert_allclose(self.results_a.smoothed_state_disturbance, self.results_b.smoothed_state_disturbance, atol=1e-07)

    def test_smoothed_measurement_disturbance(self):
        assert_allclose(self.collapse(self.results_a.smoothed_measurement_disturbance.T).T, self.results_b.smoothed_measurement_disturbance, atol=1e-07)

    def test_simulation_smoothed_measurement_disturbance(self):
        assert_allclose(self.collapse(self.sim_a.simulated_measurement_disturbance.T), self.sim_b.simulated_measurement_disturbance.T, atol=1e-07)

    def test_simulation_smoothed_state_disturbance(self):
        assert_allclose(self.sim_a.simulated_state_disturbance, self.sim_b.simulated_state_disturbance, atol=1e-07)