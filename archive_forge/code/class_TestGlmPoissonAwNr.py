import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestGlmPoissonAwNr(CheckWeight):

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs
        cls.res1 = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), var_weights=aweights).fit()
        from copy import copy
        cls.res2 = copy(res_stata.results_poisson_aweight_nonrobust)
        cls.res2.resids = cls.res2.resids.copy()
        cls.res2.resids[:, 3:5] *= np.sqrt(aweights[:, np.newaxis])