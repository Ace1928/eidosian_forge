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
class TestClark1989(Clark1989):
    """
    Basic double precision test for the loglikelihood and filtered
    states with two-dimensional observation vector.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=0)
        cls.results = cls.run_filter()

    def test_kalman_gain(self):
        assert_allclose(self.results.kalman_gain.sum(axis=1).sum(axis=0), clark1989_results['V1'], atol=0.0001)