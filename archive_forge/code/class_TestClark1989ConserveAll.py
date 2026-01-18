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
class TestClark1989ConserveAll(Clark1989):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=1 | 2 | 4 | 8)
        cls.model.loglikelihood_burn = 0
        cls.results = cls.run_filter()

    def test_loglike(self):
        assert_almost_equal(self.results.llf, self.true['loglike'], 2)

    def test_filtered_state(self):
        end = self.true_states.shape[0]
        assert_almost_equal(self.results.filtered_state[0][-1], self.true_states.iloc[end - 1, 0], 4)
        assert_almost_equal(self.results.filtered_state[1][-1], self.true_states.iloc[end - 1, 1], 4)
        assert_almost_equal(self.results.filtered_state[4][-1], self.true_states.iloc[end - 1, 2], 4)
        assert_almost_equal(self.results.filtered_state[5][-1], self.true_states.iloc[end - 1, 3], 4)