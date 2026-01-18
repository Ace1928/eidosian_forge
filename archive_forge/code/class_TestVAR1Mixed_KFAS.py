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
class TestVAR1Mixed_KFAS(CheckVAR1Mixed, CheckKFASMixin, CheckVAR1):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_var1_mixed_R.csv')

    def test_predicted_state(self):
        super().test_predicted_state(rtol_diffuse=np.inf)

    def test_filtered_state(self):
        super().test_filtered_state(rtol_diffuse=np.inf)

    def test_smoothed_state(self):
        super().test_smoothed_state(rtol_diffuse=np.inf)