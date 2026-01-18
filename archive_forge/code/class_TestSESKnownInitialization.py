import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestSESKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata)
        start_params = pd.Series([0.8, 440.0], index=mod.param_names)
        super().setup_class(mod, start_params)