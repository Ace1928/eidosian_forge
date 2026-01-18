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
class TestClark1989PartialMissing(Clark1989):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        endog = cls.model.endog
        endog[1, -51:] = np.nan
        cls.model.bind(endog)
        cls.results = cls.run_filter()

    def test_loglike(self):
        assert_allclose(self.results.llf_obs[0:].sum(), 1232.113456)

    def test_filtered_state(self):
        pass

    def test_predicted_state(self):
        assert_allclose(self.results.predicted_state.T[1:], clark1989_results.iloc[:, 1:], atol=1e-08)