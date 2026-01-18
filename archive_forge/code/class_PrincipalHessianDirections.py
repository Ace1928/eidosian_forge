import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class PrincipalHessianDirections(_DimReductionRegression):
    """
    Principal Hessian Directions (PHD)

    Parameters
    ----------
    endog : array_like (1d)
        The dependent variable
    exog : array_like (2d)
        The covariates

    Returns
    -------
    A model instance.  Call `fit` to obtain a results instance,
    from which the estimated parameters can be obtained.

    References
    ----------
    KC Li (1992).  On Principal Hessian Directions for Data
    Visualization and Dimension Reduction: Another application
    of Stein's lemma. JASA 87:420.
    """

    def fit(self, **kwargs):
        """
        Estimate the EDR space using PHD.

        Parameters
        ----------
        resid : bool, optional
            If True, use least squares regression to remove the
            linear relationship between each covariate and the
            response, before conducting PHD.

        Returns
        -------
        A results instance which can be used to access the estimated
        parameters.
        """
        resid = kwargs.get('resid', False)
        y = self.endog - self.endog.mean()
        x = self.exog - self.exog.mean(0)
        if resid:
            from statsmodels.regression.linear_model import OLS
            r = OLS(y, x).fit()
            y = r.resid
        cm = np.einsum('i,ij,ik->jk', y, x, x)
        cm /= len(y)
        cx = np.cov(x.T)
        cb = np.linalg.solve(cx, cm)
        a, b = np.linalg.eig(cb)
        jj = np.argsort(-np.abs(a))
        a = a[jj]
        params = b[:, jj]
        results = DimReductionResults(self, params, eigs=a)
        return DimReductionResultsWrapper(results)