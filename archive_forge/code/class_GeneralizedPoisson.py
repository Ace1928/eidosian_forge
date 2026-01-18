from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class GeneralizedPoisson(CountModel):
    __doc__ = '\n    Generalized Poisson Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': '\n    p : scalar\n        P denotes parameterizations for GP regression. p=1 for GP-1 and\n        p=2 for GP-2. Default is p=1.\n    offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.' + base._missing_param_doc + _check_rank_doc}

    def __init__(self, endog, exog, p=1, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, check_rank=check_rank, **kwargs)
        self.parameterization = p - 1
        self.exog_names.append('alpha')
        self.k_extra = 1
        self._transparams = False

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        kwds['p'] = self.parameterization + 1
        return kwds

    def _get_exogs(self):
        return (self.exog, None)

    def loglike(self, params):
        """
        Loglikelihood of Generalized Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[\\mu_{i}+(y_{i}-1)*ln(\\mu_{i}+
            \\alpha*\\mu_{i}^{p-1}*y_{i})-y_{i}*ln(1+\\alpha*\\mu_{i}^{p-1})-
            ln(y_{i}!)-\\frac{\\mu_{i}+\\alpha*\\mu_{i}^{p-1}*y_{i}}{1+\\alpha*
            \\mu_{i}^{p-1}}\\right]
        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generalized Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[\\mu_{i}+(y_{i}-1)*ln(\\mu_{i}+
            \\alpha*\\mu_{i}^{p-1}*y_{i})-y_{i}*ln(1+\\alpha*\\mu_{i}^{p-1})-
            ln(y_{i}!)-\\frac{\\mu_{i}+\\alpha*\\mu_{i}^{p-1}*y_{i}}{1+\\alpha*
            \\mu_{i}^{p-1}}\\right]

        for observations :math:`i=1,...,n`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        endog = self.endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + (a1 - 1) * endog
        a1 = np.maximum(1e-20, a1)
        a2 = np.maximum(1e-20, a2)
        return np.log(mu) + (endog - 1) * np.log(a2) - endog * np.log(a1) - gammaln(endog + 1) - a2 / a1

    @Appender(_get_start_params_null_docs)
    def _get_start_params_null(self):
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        mu = const * np.exp(offset + exposure)
        resid = self.endog - mu
        a = self._estimate_dispersion(mu, resid, df_resid=resid.shape[0] - 1)
        params.append(a)
        return np.array(params)

    def _estimate_dispersion(self, mu, resid, df_resid=None):
        q = self.parameterization
        if df_resid is None:
            df_resid = resid.shape[0]
        a = ((np.abs(resid) / np.sqrt(mu) - 1) * mu ** (-q)).sum() / df_resid
        return a

    @Appender('\n        use_transparams : bool\n            This parameter enable internal transformation to impose\n            non-negativity. True to enable. Default is False.\n            use_transparams=True imposes the no underdispersion (alpha > 0)\n            constraint. In case use_transparams=True and method="newton" or\n            "ncg" transformation is ignored.\n        ')
    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, use_transparams=False, cov_type='nonrobust', cov_kwds=None, use_t=None, optim_kwds_prelim=None, **kwargs):
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = True
        else:
            if use_transparams:
                warnings.warn('Parameter "use_transparams" is ignored', RuntimeWarning)
            self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            kwds_prelim = {'disp': 0, 'skip_hessian': True, 'warn_convergence': False}
            if optim_kwds_prelim is not None:
                kwds_prelim.update(optim_kwds_prelim)
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                res_poi = mod_poi.fit(**kwds_prelim)
            start_params = res_poi.params
            a = self._estimate_dispersion(res_poi.predict(), res_poi.resid, df_resid=res_poi.df_resid)
            start_params = np.append(start_params, max(-0.1, a))
        if callback is None:
            callback = lambda *x: x
        mlefit = super().fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, full_output=full_output, callback=callback, **kwargs)
        if optim_kwds_prelim is not None:
            mlefit.mle_settings['optim_kwds_prelim'] = optim_kwds_prelim
        if use_transparams and method not in ['newton', 'ncg']:
            self._transparams = False
            mlefit._results.params[-1] = np.exp(mlefit._results.params[-1])
        gpfit = GeneralizedPoissonResults(self, mlefit._results)
        result = GeneralizedPoissonResultsWrapper(gpfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        _validate_l1_method(method)
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1] + self.k_extra
            alpha = alpha * np.ones(k_params)
            alpha[-1] = 0
        alpha_p = alpha[:-1] if self.k_extra and np.size(alpha) > 1 else alpha
        self._transparams = False
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            mod_poi = Poisson(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                start_params = mod_poi.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
            start_params = np.append(start_params, 0.1)
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1GeneralizedPoissonResults(self, cntfit)
        return L1GeneralizedPoissonResultsWrapper(discretefit)

    def score_obs(self, params):
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        dmudb = mu * exog
        dalpha = mu_p * (y * ((y - 1) / a2 - 2 / a1) + a2 / a1 ** 2)
        dparams = dmudb * (-a4 / a1 + a3 * a2 / a1 ** 2 + (1 + a4) * ((y - 1) / a2 - 1 / a1) + 1 / mu)
        return np.concatenate((dparams, np.atleast_2d(dalpha)), axis=1)

    def score(self, params):
        score = np.sum(self.score_obs(params), axis=0)
        if self._transparams:
            score[-1] == score[-1] ** 2
            return score
        else:
            return score

    def score_factor(self, params, endog=None):
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog if endog is None else endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        dmudb = mu
        dalpha = mu_p * (y * ((y - 1) / a2 - 2 / a1) + a2 / a1 ** 2)
        dparams = dmudb * (-a4 / a1 + a3 * a2 / a1 ** 2 + (1 + a4) * ((y - 1) / a2 - 1 / a1) + 1 / mu)
        return (dparams, dalpha)

    def _score_p(self, params):
        """
        Generalized Poisson model derivative of the log-likelihood by p-parameter

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        dldp : float
            dldp is first derivative of the loglikelihood function,
        evaluated at `p-parameter`.
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        dp = np.sum(np.log(mu) * ((a2 - mu) * ((y - 1) / a2 - 2 / a1) + (a1 - 1) * a2 / a1 ** 2))
        return dp

    def hessian(self, params):
        """
        Generalized Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array_like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`
        """
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        exog = self.exog
        y = self.endog[:, None]
        mu = self.predict(params)[:, None]
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        a5 = p * mu ** (p - 1)
        dmudb = mu * exog
        dim = exog.shape[1]
        hess_arr = np.empty((dim + 1, dim + 1))
        for i in range(dim):
            for j in range(i + 1):
                hess_val = np.sum(mu * exog[:, i, None] * exog[:, j, None] * (mu * (a3 * a4 / a1 ** 2 - 2 * a3 ** 2 * a2 / a1 ** 3 + 2 * a3 * (a4 + 1) / a1 ** 2 - a4 * p / (mu * a1) + a3 * p * a2 / (mu * a1 ** 2) + (y - 1) * a4 * (p - 1) / (a2 * mu) - (y - 1) * (1 + a4) ** 2 / a2 ** 2 - a4 * (p - 1) / (a1 * mu)) + ((y - 1) * (1 + a4) / a2 - (1 + a4) / a1)), axis=0)
                hess_arr[i, j] = np.squeeze(hess_val)
        tri_idx = np.triu_indices(dim, k=1)
        hess_arr[tri_idx] = hess_arr.T[tri_idx]
        dldpda = np.sum((2 * a4 * mu_p / a1 ** 2 - 2 * a3 * mu_p * a2 / a1 ** 3 - mu_p * y * (y - 1) * (1 + a4) / a2 ** 2 + mu_p * (1 + a4) / a1 ** 2 + a5 * y * (y - 1) / a2 - 2 * a5 * y / a1 + a5 * a2 / a1 ** 2) * dmudb, axis=0)
        hess_arr[-1, :-1] = dldpda
        hess_arr[:-1, -1] = dldpda
        dldada = mu_p ** 2 * (3 * y / a1 ** 2 - (y / a2) ** 2.0 * (y - 1) - 2 * a2 / a1 ** 3)
        hess_arr[-1, -1] = dldada.sum()
        return hess_arr

    def hessian_factor(self, params):
        """
        Generalized Poisson model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (nobs, 3)
            The Hessian factor, second derivative of loglikelihood function
            with respect to linear predictor and dispersion parameter
            evaluated at `params`
            The first column contains the second derivative w.r.t. linpred,
            the second column contains the cross derivative, and the
            third column contains the second derivative w.r.t. the dispersion
            parameter.

        """
        params = np.asarray(params)
        if self._transparams:
            alpha = np.exp(params[-1])
        else:
            alpha = params[-1]
        params = params[:-1]
        p = self.parameterization
        y = self.endog
        mu = self.predict(params)
        mu_p = np.power(mu, p)
        a1 = 1 + alpha * mu_p
        a2 = mu + alpha * mu_p * y
        a3 = alpha * p * mu ** (p - 1)
        a4 = a3 * y
        a5 = p * mu ** (p - 1)
        dmudb = mu
        dbb = mu * (mu * (a3 * a4 / a1 ** 2 - 2 * a3 ** 2 * a2 / a1 ** 3 + 2 * a3 * (a4 + 1) / a1 ** 2 - a4 * p / (mu * a1) + a3 * p * a2 / (mu * a1 ** 2) + a4 / (mu * a1) - a3 * a2 / (mu * a1 ** 2) + (y - 1) * a4 * (p - 1) / (a2 * mu) - (y - 1) * (1 + a4) ** 2 / a2 ** 2 - a4 * (p - 1) / (a1 * mu) - 1 / mu ** 2) + (-a4 / a1 + a3 * a2 / a1 ** 2 + (y - 1) * (1 + a4) / a2 - (1 + a4) / a1 + 1 / mu))
        dba = (2 * a4 * mu_p / a1 ** 2 - 2 * a3 * mu_p * a2 / a1 ** 3 - mu_p * y * (y - 1) * (1 + a4) / a2 ** 2 + mu_p * (1 + a4) / a1 ** 2 + a5 * y * (y - 1) / a2 - 2 * a5 * y / a1 + a5 * a2 / a1 ** 2) * dmudb
        daa = mu_p ** 2 * (3 * y / a1 ** 2 - (y / a2) ** 2.0 * (y - 1) - 2 * a2 / a1 ** 3)
        return (dbb, dba, daa)

    @Appender(Poisson.predict.__doc__)
    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', y_values=None):
        if exog is None:
            exog = self.exog
        if exposure is None:
            exposure = getattr(self, 'exposure', 0)
        elif exposure != 0:
            exposure = np.log(exposure)
        if offset is None:
            offset = getattr(self, 'offset', 0)
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which == 'linear':
            return linpred
        elif which == 'var':
            mean = np.exp(linpred)
            alpha = params[-1]
            pm1 = self.parameterization
            var_ = mean * (1 + alpha * mean ** pm1) ** 2
            return var_
        elif which == 'prob':
            if y_values is None:
                y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)[:, None]
            return genpoisson_p.pmf(y_values, mu, params[-1], self.parameterization + 1)
        else:
            raise ValueError("keyword 'which' not recognized")

    def _deriv_score_obs_dendog(self, params):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)
        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2]
        return np.column_stack((d1, d2))

    def _var(self, mu, params=None):
        """variance implied by the distribution

        internal use, will be refactored or removed
        """
        alpha = params[-1]
        pm1 = self.parameterization
        var_ = mu * (1 + alpha * mu ** pm1) ** 2
        return var_

    def _prob_nonzero(self, mu, params):
        """Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
        alpha = params[-1]
        pm1 = self.parameterization
        prob_zero = np.exp(-mu / (1 + alpha * mu ** pm1))
        prob_nz = 1 - prob_zero
        return prob_nz

    @Appender(Poisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exposure=None, offset=None):
        """get frozen instance of distribution
        """
        mu = self.predict(params, exog=exog, exposure=exposure, offset=offset)
        p = self.parameterization + 1
        distr = genpoisson_p(mu, params[-1], p)
        return distr