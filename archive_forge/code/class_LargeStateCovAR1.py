import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
class LargeStateCovAR1(mlemodel.MLEModel):
    """
    Test class for k_posdef > k_states (which usually do not get tested in
    other models).

    This is just an AR(1) model with an extra unused state innovation
    """

    def __init__(self, endog, **kwargs):
        k_states = 1
        k_posdef = 2
        super().__init__(endog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        self['design', 0, 0] = 1
        self['selection', 0, 0] = 1
        self['state_cov', 1, 1] = 1
        self.initialize_stationary()

    @property
    def param_names(self):
        return ['phi', 'sigma2']

    @property
    def start_params(self):
        return [0.5, 1]

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self['transition', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]