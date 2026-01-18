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
class TestGenericGLMPoissonOffset(CheckGenericMixin):

    def setup_method(self):
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.GLM(y_count, x, family=sm.families.Poisson(), offset=0.01 * np.ones(nobs), exposure=np.ones(nobs))
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)
        self.predict_kwds_5 = dict(exposure=0.01 * np.ones(5), offset=np.ones(5))
        self.predict_kwds = dict(exposure=1, offset=0)