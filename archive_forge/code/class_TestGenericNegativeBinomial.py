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
class TestGenericNegativeBinomial(CheckGenericMixin):

    def setup_method(self):
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        data.exog = np.asarray(data.exog)
        data.endog = np.asarray(data.endog)
        exog = sm.add_constant(data.exog, prepend=False)
        mod = sm.NegativeBinomial(data.endog, exog)
        start_params = np.array([-0.05783623, -0.26655806, 0.04109148, -0.03815837, 0.2685168, 0.03811594, -0.04426238, 0.01614795, 0.17490962, 0.66461151, 1.2925957])
        self.results = mod.fit(start_params=start_params, disp=0, maxiter=500)
        self.transform_index = -1