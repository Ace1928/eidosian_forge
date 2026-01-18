import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
def _get_start(self):
    model = OLS(self.endog, self.exog)
    result = model.fit()
    m = self.exog_scale.shape[1] + self.exog_smooth.shape[1]
    if self._has_noise:
        m += self.exog_noise.shape[1]
    return np.concatenate((result.params, np.zeros(m)))