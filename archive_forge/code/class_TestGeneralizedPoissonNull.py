from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestGeneralizedPoissonNull(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = GeneralizedPoisson(endog, exog, p=1.5)
        cls.model_null = GeneralizedPoisson(endog, exog[:, 0], p=1.5)
        cls.res_null = cls.model_null.fit(start_params=[8.4, 1], method='bfgs', gtol=1e-08, maxiter=300, disp=0)
        cls.start_params = np.array([6.91127148, 0.04501334, 0.88393736])