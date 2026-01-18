import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
class TestStatesAR3AlternativeSmoothing(TestStatesAR3):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(*args, smooth_method=SMOOTH_ALTERNATIVE, **kwargs)

    def test_smoothed_states(self):
        assert_almost_equal(self.results.smoother_results.smoothed_state.T[2:], self.stata.iloc[3:][['sm1', 'sm2', 'sm3']], 4)
        assert_almost_equal(self.results.smoother_results.smoothed_state.T[2:], self.matlab_ssm.iloc[2:][['alphahat1', 'alphahat2', 'alphahat3']], 4)

    def test_smoothed_states_cov(self):
        assert_almost_equal(self.results.det_smoothed_state_cov.T[1:], self.matlab_ssm.iloc[1:][['detV']], 4)

    def test_smooth_method(self):
        assert_equal(self.model.ssm.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother.smooth_method, SMOOTH_ALTERNATIVE)
        assert_equal(self.model.ssm._kalman_smoother._smooth_method, SMOOTH_ALTERNATIVE)