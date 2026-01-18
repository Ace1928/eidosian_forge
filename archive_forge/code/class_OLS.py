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
class OLS(WLS):
    __doc__ = '\n    Ordinary Least Squares\n\n    {params}\n    {extra_params}\n\n    Attributes\n    ----------\n    weights : scalar\n        Has an attribute weights = array(1.0) due to inheritance from WLS.\n\n    See Also\n    --------\n    WLS : Fit a linear model using Weighted Least Squares.\n    GLS : Fit a linear model using Generalized Least Squares.\n\n    Notes\n    -----\n    No constant is added by the model unless you are using formulas.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> import numpy as np\n    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")\n    >>> Y = duncan_prestige.data[\'income\']\n    >>> X = duncan_prestige.data[\'education\']\n    >>> X = sm.add_constant(X)\n    >>> model = sm.OLS(Y,X)\n    >>> results = model.fit()\n    >>> results.params\n    const        10.603498\n    education     0.594859\n    dtype: float64\n\n    >>> results.tvalues\n    const        2.039813\n    education    6.892802\n    dtype: float64\n\n    >>> print(results.t_test([1, 0]))\n                                 Test for Constraints\n    ==============================================================================\n                     coef    std err          t      P>|t|      [0.025      0.975]\n    ------------------------------------------------------------------------------\n    c0            10.6035      5.198      2.040      0.048       0.120      21.087\n    ==============================================================================\n\n    >>> print(results.f_test(np.identity(2)))\n    <F test: F=array([[159.63031026]]), p=1.2607168903696672e-20,\n     df_denom=43, df_num=2>\n    '.format(params=base._model_params_doc, extra_params=base._missing_param_doc + base._extra_param_doc)

    def __init__(self, endog, exog=None, missing='none', hasconst=None, **kwargs):
        if 'weights' in kwargs:
            msg = 'Weights are not supported in OLS and will be ignoredAn exception will be raised in the next version.'
            warnings.warn(msg, ValueWarning)
        super().__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)
        if 'weights' in self._init_keys:
            self._init_keys.remove('weights')
        if type(self) is OLS:
            self._check_kwargs(kwargs, ['offset'])

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        float
            The likelihood function evaluated at params.
        """
        nobs2 = self.nobs / 2.0
        nobs = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        if hasattr(self, 'offset'):
            resid -= self.offset
        ssr = np.sum(resid ** 2)
        if scale is None:
            llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
        else:
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2 * scale)
        return llf

    def whiten(self, x):
        """
        OLS model whitener does nothing.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The input array unmodified.

        See Also
        --------
        OLS : Fit a linear model using Ordinary Least Squares.
        """
        return x

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The score vector.
        """
        if not hasattr(self, '_wexog_xprod'):
            self._setup_score_hess()
        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb
        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T, params)
            ssr += np.dot(params, xtxb)
            return -self.nobs * sdr / ssr
        else:
            return -sdr / scale

    def _setup_score_hess(self):
        y = self.wendog
        if hasattr(self, 'offset'):
            y = y - self.offset
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog = np.dot(self.wexog.T, y)

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The Hessian matrix.
        """
        if not hasattr(self, '_wexog_xprod'):
            self._setup_score_hess()
        xtxb = np.dot(self._wexog_xprod, params)
        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T, params)
            ssr += np.dot(params, xtxb)
            ssrp = -2 * self._wexog_x_wendog + 2 * xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr ** 2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Calculate the weights for the Hessian.

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
        return np.ones(self.exog.shape[0])

    @Appender(_fit_regularized_doc)
    def fit_regularized(self, method='elastic_net', alpha=0.0, L1_wt=1.0, start_params=None, profile_scale=False, refit=False, **kwargs):
        if method not in ('elastic_net', 'sqrt_lasso'):
            msg = "Unknown method '%s' for fit_regularized" % method
            raise ValueError(msg)
        defaults = {'maxiter': 50, 'cnvrg_tol': 1e-10, 'zero_tol': 1e-08}
        defaults.update(kwargs)
        if method == 'sqrt_lasso':
            from statsmodels.base.elastic_net import RegularizedResults, RegularizedResultsWrapper
            params = self._sqrt_lasso(alpha, refit, defaults['zero_tol'])
            results = RegularizedResults(self, params)
            return RegularizedResultsWrapper(results)
        from statsmodels.base.elastic_net import fit_elasticnet
        if L1_wt == 0:
            return self._fit_ridge(alpha)
        if profile_scale:
            loglike_kwds = {}
            score_kwds = {}
            hess_kwds = {}
        else:
            loglike_kwds = {'scale': 1}
            score_kwds = {'scale': 1}
            hess_kwds = {'scale': 1}
        return fit_elasticnet(self, method=method, alpha=alpha, L1_wt=L1_wt, start_params=start_params, loglike_kwds=loglike_kwds, score_kwds=score_kwds, hess_kwds=hess_kwds, refit=refit, check_step=False, **defaults)

    def _sqrt_lasso(self, alpha, refit, zero_tol):
        try:
            import cvxopt
        except ImportError:
            msg = 'sqrt_lasso fitting requires the cvxopt module'
            raise ValueError(msg)
        n = len(self.endog)
        p = self.exog.shape[1]
        h0 = cvxopt.matrix(0.0, (2 * p + 1, 1))
        h1 = cvxopt.matrix(0.0, (n + 1, 1))
        h1[1:, 0] = cvxopt.matrix(self.endog, (n, 1))
        G0 = cvxopt.spmatrix([], [], [], (2 * p + 1, 2 * p + 1))
        for i in range(1, 2 * p + 1):
            G0[i, i] = -1
        G1 = cvxopt.matrix(0.0, (n + 1, 2 * p + 1))
        G1[0, 0] = -1
        G1[1:, 1:p + 1] = self.exog
        G1[1:, p + 1:] = -self.exog
        c = cvxopt.matrix(alpha / n, (2 * p + 1, 1))
        c[0] = 1 / np.sqrt(n)
        from cvxopt import solvers
        solvers.options['show_progress'] = False
        rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
        x = np.asarray(rslt['x']).flat
        bp = x[1:p + 1]
        bn = x[p + 1:]
        params = bp - bn
        if not refit:
            return params
        ii = np.flatnonzero(np.abs(params) > zero_tol)
        rfr = OLS(self.endog, self.exog[:, ii]).fit()
        params *= 0
        params[ii] = rfr.params
        return params

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """
        u, s, vt = np.linalg.svd(self.exog, 0)
        v = vt.T
        q = np.dot(u.T, self.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * self.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)
        from statsmodels.base.elastic_net import RegularizedResults
        return RegularizedResults(self, params)