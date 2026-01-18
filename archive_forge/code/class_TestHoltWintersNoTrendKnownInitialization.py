import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltWintersNoTrendKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series([0.5, 0.49, 30.0, 2.0, -2, -9], index=mod.param_names)
        super().setup_class(mod, start_params)