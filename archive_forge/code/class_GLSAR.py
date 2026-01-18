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
class GLSAR(GLS):
    __doc__ = '\n    Generalized Least Squares with AR covariance structure\n\n    {params}\n    rho : int\n        The order of the autoregressive covariance.\n    {extra_params}\n\n    Notes\n    -----\n    GLSAR is considered to be experimental.\n    The linear autoregressive process of order p--AR(p)--is defined as:\n    TODO\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> X = range(1,8)\n    >>> X = sm.add_constant(X)\n    >>> Y = [1,3,4,5,8,10,9]\n    >>> model = sm.GLSAR(Y, X, rho=2)\n    >>> for i in range(6):\n    ...     results = model.fit()\n    ...     print("AR coefficients: {{0}}".format(model.rho))\n    ...     rho, sigma = sm.regression.yule_walker(results.resid,\n    ...                                            order=model.order)\n    ...     model = sm.GLSAR(Y, X, rho)\n    ...\n    AR coefficients: [ 0.  0.]\n    AR coefficients: [-0.52571491 -0.84496178]\n    AR coefficients: [-0.6104153  -0.86656458]\n    AR coefficients: [-0.60439494 -0.857867  ]\n    AR coefficients: [-0.6048218  -0.85846157]\n    AR coefficients: [-0.60479146 -0.85841922]\n    >>> results.params\n    array([-0.66661205,  1.60850853])\n    >>> results.tvalues\n    array([ -2.10304127,  21.8047269 ])\n    >>> print(results.t_test([1, 0]))\n    <T test: effect=array([-0.66661205]), sd=array([[ 0.31697526]]),\n     t=array([[-2.10304127]]), p=array([[ 0.06309969]]), df_denom=3>\n    >>> print(results.f_test(np.identity(2)))\n    <F test: F=array([[ 1815.23061844]]), p=[[ 0.00002372]],\n     df_denom=3, df_num=2>\n\n    Or, equivalently\n\n    >>> model2 = sm.GLSAR(Y, X, rho=2)\n    >>> res = model2.iterative_fit(maxiter=6)\n    >>> model2.rho\n    array([-0.60479146, -0.85841922])\n    '.format(params=base._model_params_doc, extra_params=base._missing_param_doc + base._extra_param_doc)

    def __init__(self, endog, exog=None, rho=1, missing='none', hasconst=None, **kwargs):
        if isinstance(rho, (int, np.integer)):
            self.order = int(rho)
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0, 1]:
                raise ValueError('AR parameters must be a scalar or a vector')
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        if exog is None:
            super().__init__(endog, np.ones((endog.shape[0], 1)), missing=missing, hasconst=None, **kwargs)
        else:
            super().__init__(endog, exog, missing=missing, **kwargs)

    def iterative_fit(self, maxiter=3, rtol=0.0001, **kwargs):
        """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated iteratively.

        Parameters
        ----------
        maxiter : int, optional
            The number of iterations.
        rtol : float, optional
            Relative tolerance between estimated coefficients to stop the
            estimation.  Stops if max(abs(last - current) / abs(last)) < rtol.
        **kwargs
            Additional keyword arguments passed to `fit`.

        Returns
        -------
        RegressionResults
            The results computed using an iterative fit.
        """
        converged = False
        i = -1
        history = {'params': [], 'rho': [self.rho]}
        for i in range(maxiter - 1):
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            self.initialize()
            results = self.fit()
            history['params'].append(results.params)
            if i == 0:
                last = results.params
            else:
                diff = np.max(np.abs(last - results.params) / np.abs(last))
                if diff < rtol:
                    converged = True
                    break
                last = results.params
            self.rho, _ = yule_walker(results.resid, order=self.order, df=None)
            history['rho'].append(self.rho)
        if not converged and maxiter > 0:
            if hasattr(self, 'pinv_wexog'):
                del self.pinv_wexog
            self.initialize()
        results = self.fit(history=history, **kwargs)
        results.iter = i + 1
        if not converged:
            results.history['params'].append(results.params)
            results.iter += 1
        results.converged = converged
        return results

    def whiten(self, x):
        """
        Whiten a series of columns according to an AR(p) covariance structure.

        Whitening using this method drops the initial p observations.

        Parameters
        ----------
        x : array_like
            The data to be whitened.

        Returns
        -------
        ndarray
            The whitened data.
        """
        x = np.asarray(x, np.float64)
        _x = x.copy()
        for i in range(self.order):
            _x[i + 1:] = _x[i + 1:] - self.rho[i] * x[0:-(i + 1)]
        return _x[self.order:]