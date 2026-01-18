import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestSESHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method='heuristic')
        super().setup_class(mod)

    def test_heuristic(self):
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(oildata.values[:nobs])[0]
        assert_allclose(self.init_heuristic, desired)