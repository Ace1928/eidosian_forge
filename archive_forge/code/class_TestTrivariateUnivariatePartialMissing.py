import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
class TestTrivariateUnivariatePartialMissing(Trivariate):

    @classmethod
    def setup_class(cls, dtype=float, **kwargs):
        super().setup_class(dtype, **kwargs)
        nobs = cls.model.nobs
        k_endog = cls.model.k_endog
        k_posdef = cls.model.k_posdef
        cls.model.endog[:2, 10:180] = np.nan
        cls.model.filter_univariate = True
        cls.model.filter_collapsed = True
        cls.results_b = cls.model.smooth()
        cls.sim_b = cls.model.simulation_smoother(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(cls.model.k_states))
        cls.model.filter_collapsed = False
        cls.results_a = cls.model.smooth()
        cls.sim_a = cls.model.simulation_smoother(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(cls.model.k_states))