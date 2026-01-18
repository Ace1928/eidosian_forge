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
class CheckVAR1Missing(CheckVAR1):

    @classmethod
    def setup_class(cls, **kwargs):
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[8:12, :] = np.nan
        kwargs['endog'] = endog
        super().setup_class(**kwargs)

    def test_nobs_diffuse(self):
        assert_allclose(self.d, 2)