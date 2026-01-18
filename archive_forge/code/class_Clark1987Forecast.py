import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
class Clark1987Forecast(Clark1987):
    """
    Forecasting test for the loglikelihood and filtered states.
    """

    @classmethod
    def setup_class(cls, dtype=float, nforecast=100, conserve_memory=0):
        super().setup_class(dtype=dtype, conserve_memory=conserve_memory)
        cls.nforecast = nforecast
        cls.model.endog = np.array(np.r_[cls.model.endog[0, :], [np.nan] * nforecast], ndmin=2, dtype=dtype, order='F')
        cls.model.nobs = cls.model.endog.shape[1]

    def test_filtered_state(self):
        assert_almost_equal(self.results.filtered_state[0][self.true['start']:-self.nforecast], self.true_states.iloc[:, 0], 4)
        assert_almost_equal(self.results.filtered_state[1][self.true['start']:-self.nforecast], self.true_states.iloc[:, 1], 4)
        assert_almost_equal(self.results.filtered_state[3][self.true['start']:-self.nforecast], self.true_states.iloc[:, 2], 4)