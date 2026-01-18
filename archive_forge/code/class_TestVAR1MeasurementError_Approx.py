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
class TestVAR1MeasurementError_Approx(CheckApproximateDiffuseMixin, CheckVAR1MeasurementError):
    approximate_diffuse_variance = 1000000000.0

    def test_smoothed_measurement_disturbance_cov(self, rtol_diffuse=None):
        super().test_smoothed_measurement_disturbance_cov(rtol_diffuse=rtol_diffuse)