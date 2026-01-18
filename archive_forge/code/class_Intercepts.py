import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
class Intercepts(mlemodel.MLEModel):
    """
    Test class for observation and state intercepts (which usually do not
    get tested in other models).
    """

    def __init__(self, endog, **kwargs):
        k_states = 3
        k_posdef = 3
        super().__init__(endog, k_states=k_states, k_posdef=k_posdef, **kwargs)
        self['design'] = np.eye(3)
        self['obs_cov'] = np.eye(3)
        self['transition'] = np.eye(3)
        self['selection'] = np.eye(3)
        self['state_cov'] = np.eye(3)
        self.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return ['d.1', 'd.2', 'd.3', 'c.1', 'c.2', 'c.3']

    @property
    def start_params(self):
        return np.arange(6)

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self['obs_intercept'] = params[:3]
        self['state_intercept'] = params[3:]