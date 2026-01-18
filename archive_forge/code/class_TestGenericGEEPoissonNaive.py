from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
class TestGenericGEEPoissonNaive(CheckGenericMixin):

    def setup_method(self):
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])
        start_params = np.array([0.0, 1.0, 1.0, 1.0])
        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi).fit(start_params=start_params, cov_type='naive')