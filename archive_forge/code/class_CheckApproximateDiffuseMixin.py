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
class CheckApproximateDiffuseMixin:
    """
    Test the exact diffuse initialization against the approximate diffuse
    initialization. By definition, the first few observations will be quite
    different between the exact and approximate approach for many quantities,
    so we do not test them here.
    """
    approximate_diffuse_variance = 1000000.0

    @classmethod
    def setup_class(cls, *args, **kwargs):
        init_approx = kwargs.pop('init_approx', None)
        super().setup_class(*args, **kwargs)
        kappa = cls.approximate_diffuse_variance
        if init_approx is None:
            init_approx = Initialization(cls.ssm.k_states, 'approximate_diffuse', approximate_diffuse_variance=kappa)
        cls.ssm.initialize(init_approx)
        cls.results_b = cls.ssm.smooth()
        cls.rtol_diffuse = np.inf

    def test_initialization_approx(self):
        kappa = self.approximate_diffuse_variance
        assert_allclose(self.results_b.initial_state_cov, np.eye(self.ssm.k_states) * kappa)
        assert_equal(self.results_b.initial_diffuse_state_cov, None)