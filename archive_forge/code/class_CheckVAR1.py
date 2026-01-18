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
class CheckVAR1(CheckSSMResults):

    @classmethod
    def setup_class(cls, **kwargs):
        filter_univariate = kwargs.pop('filter_univariate', False)
        cls.mod, cls.ssm = model_var1(**kwargs)
        if filter_univariate:
            cls.ssm.filter_univariate = True
        cls.results_a = cls.ssm.smooth()
        cls.d = cls.results_a.nobs_diffuse

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 1)

    def test_initialization(self):
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))