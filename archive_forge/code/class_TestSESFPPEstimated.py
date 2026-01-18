import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestSESFPPEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method='estimated', concentrate_scale=False)
        res = mod.filter([results_params['oil_fpp3']['alpha'], results_params['oil_fpp3']['sigma2'], results_params['oil_fpp3']['l0']])
        super().setup_class('oil_fpp3', res)