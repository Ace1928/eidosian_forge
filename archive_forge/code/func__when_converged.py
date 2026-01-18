import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def _when_converged(self, atol=1e-08, rtol=0, tol_criterion='deviance'):
    for i, dev in enumerate(self.res.fit_history[tol_criterion]):
        orig = self.res.fit_history[tol_criterion][i]
        new = self.res.fit_history[tol_criterion][i + 1]
        if np.allclose(orig, new, atol=atol, rtol=rtol):
            return i
    raise ValueError("CONVERGENCE CHECK: It seems this doens't converge!")