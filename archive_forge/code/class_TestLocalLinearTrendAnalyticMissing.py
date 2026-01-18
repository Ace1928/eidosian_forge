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
class TestLocalLinearTrendAnalyticMissing(TestLocalLinearTrendAnalytic):

    @classmethod
    def setup_class(cls):
        y1 = 10.2394
        y2 = np.nan
        y3 = 6.123123
        endog = np.r_[y1, y2, y3, [1] * 7]
        super().setup_class(endog=endog)

    def test_results(self):
        ssm = self.ssm
        res = self.res
        y1, y2, y3 = ssm.endog[0, :3]
        sigma2_y = ssm['obs_cov', 0, 0]
        sigma2_mu, sigma2_beta = np.diagonal(ssm['state_cov'])
        q_mu = sigma2_mu / sigma2_y
        q_beta = sigma2_beta / sigma2_y
        a4 = [1.5 * y3 - 0.5 * y1, 0.5 * y3 - 0.5 * y1]
        assert_allclose(res.predicted_state[:, 3], a4)
        P4 = sigma2_y * np.array([[2.5 + 1.5 * q_mu + 1.25 * q_beta, 1 + 0.5 * q_mu + 1.25 * q_beta], [1 + 0.5 * q_mu + 1.25 * q_beta, 0.5 + 0.5 * q_mu + 2.25 * q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 3], P4)
        assert_equal(res.nobs_diffuse, 3)