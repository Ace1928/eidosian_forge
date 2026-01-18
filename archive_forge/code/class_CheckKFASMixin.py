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
class CheckKFASMixin:
    """
    Test against values from KFAS
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs.setdefault('filter_univariate', True)
        super().setup_class(*args, **kwargs)
        cls.results_b = kfas_helpers.parse(cls.results_path, cls.ssm)
        cls.results_b.smoothed_state_autocov = None
        cls.results_b.kalman_gain = None
        cls.results_b.filtered_state_cov = None
        Finf = cls.results_b.forecasts_error_diffuse_cov.T
        Finf_nonsingular_obs = np.c_[[np.diag(Finf_t) for Finf_t in Finf]] > 0
        nonmissing = ~np.isnan(cls.ssm.endog).T
        constant = -0.5 * np.log(2 * np.pi) * (Finf_nonsingular_obs * nonmissing).sum(axis=1)
        cls.results_b.llf_obs += constant[:cls.results_a.nobs_diffuse].sum()