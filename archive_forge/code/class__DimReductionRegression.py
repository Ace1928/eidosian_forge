import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class _DimReductionRegression(model.Model):
    """
    A base class for dimension reduction regression methods.
    """

    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog, exog, **kwargs)

    def _prep(self, n_slice):
        ii = np.argsort(self.endog)
        x = self.exog[ii, :]
        x -= x.mean(0)
        covx = np.dot(x.T, x) / x.shape[0]
        covxr = np.linalg.cholesky(covx)
        x = np.linalg.solve(covxr, x.T).T
        self.wexog = x
        self._covxr = covxr
        self._split_wexog = np.array_split(x, n_slice)