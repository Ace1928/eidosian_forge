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
class TestClark1989ForecastConserve(Clark1989Forecast):
    """
    Memory conservation forecasting test for the loglikelihood and filtered
    states.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class(dtype=float, conserve_memory=1 | 2)
        cls.results = cls.run_filter()