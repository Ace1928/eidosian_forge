import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersDampedConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True, seasonal=4)
        start_params = pd.Series([0.0005, 0.0004, 0.0005, 0.95, 17.0, 1.5, -0.2, 0.1, 0.4], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=0.1)