from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
class WLS(RegressionModel):
    __doc__ = '\n    Weighted Least Squares\n\n    The weights are presumed to be (proportional to) the inverse of\n    the variance of the observations.  That is, if the variables are\n    to be transformed by 1/sqrt(W) you must supply weights = 1/W.\n\n    {params}\n    weights : array_like, optional\n        A 1d array of weights.  If you supply 1/W then the variables are\n        pre- multiplied by 1/sqrt(W).  If no weights are supplied the\n        default value is 1 and WLS results are the same as OLS.\n    {extra_params}\n\n    Attributes\n    ----------\n    weights : ndarray\n        The stored weights supplied as an argument.\n\n    See Also\n    --------\n    GLS : Fit a linear model using Generalized Least Squares.\n    OLS : Fit a linear model using Ordinary Least Squares.\n\n    Notes\n    -----\n    If the weights are a function of the data, then the post estimation\n    statistics such as fvalue and mse_model might not be correct, as the\n    package does not yet support no-constant regression.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> Y = [1,3,4,5,2,3,4]\n    >>> X = range(1,8)\n    >>> X = sm.add_constant(X)\n    >>> wls_model = sm.WLS(Y,X, weights=list(range(1,8)))\n    >>> results = wls_model.fit()\n    >>> results.params\n    array([ 2.91666667,  0.0952381 ])\n    >>> results.tvalues\n    array([ 2.0652652 ,  0.35684428])\n    >>> print(results.t_test([1, 0]))\n    <T test: effect=array([ 2.91666667]), sd=array([[ 1.41224801]]),\n     t=array([[ 2.0652652]]), p=array([[ 0.04690139]]), df_denom=5>\n    >>> print(results.f_test([0, 1]))\n    <F test: F=array([[ 0.12733784]]), p=[[ 0.73577409]], df_denom=5, df_num=1>\n    '.format(params=base._model_params_doc, extra_params=base._missing_param_doc + base._extra_param_doc)

    def __init__(self, endog, exog, weights=1.0, missing='none', hasconst=None, **kwargs):
        if type(self) is WLS:
            self._check_kwargs(kwargs)
        weights = np.array(weights)
        if weights.shape == ():
            if missing == 'drop' and 'missing_idx' in kwargs and (kwargs['missing_idx'] is not None):
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super().__init__(endog, exog, missing=missing, weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, x):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights).

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The whitened values sqrt(weights)*X.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return x * np.sqrt(self.weights)
        elif x.ndim == 2:
            return np.sqrt(self.weights)[:, None] * x

    def loglike(self, params):
        """
        Compute the value of the gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array_like
            The parameter estimates.

        Returns
        -------
        float
            The value of the log-likelihood function for a WLS Model.

        Notes
        -----
        .. math:: -\\frac{n}{2}\\log SSR
                  -\\frac{n}{2}\\left(1+\\log\\left(\\frac{2\\pi}{n}\\right)\\right)
                  +\\frac{1}{2}\\log\\left(\\left|W\\right|\\right)

        where :math:`W` is a diagonal weight matrix,
        :math:`\\left|W\\right|` is its determinant, and
        :math:`SSR=\\left(Y-\\hat{Y}\\right)^\\prime W \\left(Y-\\hat{Y}\\right)` is
        the sum of the squared weighted residuals.
        """
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params)) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        llf += 0.5 * np.sum(np.log(self.weights))
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute the weights for calculating the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        return self.weights

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method='elastic_net', alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False, **kwargs):
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        alpha = alpha * np.sum(self.weights) / len(self.weights)
        rslt = OLS(self.wendog, self.wexog).fit_regularized(method=method, alpha=alpha, L1_wt=L1_wt, start_params=start_params, profile_scale=profile_scale, refit=refit, **kwargs)
        from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)