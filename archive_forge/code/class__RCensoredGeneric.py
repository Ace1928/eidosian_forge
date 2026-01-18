import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import (
from statsmodels.discrete.discrete_model import (
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy
class _RCensoredGeneric(CountModel):
    __doc__ = '\n    Generic right Censored model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none', **kwargs):
        self.zero_idx = np.nonzero(endog == 0)[0]
        self.nonzero_idx = np.nonzero(endog)[0]
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)

    def loglike(self, params):
        """
        Loglikelihood of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        -----

        """
        return np.sum(self.loglikeobs(params))

    def loglikeobs(self, params):
        """
        Loglikelihood for observations of Generic Censored model

        Parameters
        ----------
        params : array-like
            The parameters of the model.

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        -----

        """
        llf_main = self.model_main.loglikeobs(params)
        llf = np.concatenate((llf_main[self.zero_idx], np.log(1 - np.exp(llf_main[self.nonzero_idx]))))
        return llf

    def score_obs(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        score_main = self.model_main.score_obs(params)
        llf_main = self.model_main.loglikeobs(params)
        score = np.concatenate((score_main[self.zero_idx], (score_main[self.nonzero_idx].T * -np.exp(llf_main[self.nonzero_idx]) / (1 - np.exp(llf_main[self.nonzero_idx]))).T))
        return score

    def score(self, params):
        """
        Generic Censored model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`
        """
        return self.score_obs(params).sum(0)

    def fit(self, start_params=None, method='bfgs', maxiter=35, full_output=1, disp=1, callback=None, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=ConvergenceWarning)
                start_params = model.fit(disp=0).params
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, disp=disp, full_output=full_output, callback=lambda x: x, **kwargs)
        zipfit = self.result_class(self, mlefit._results)
        result = self.result_class_wrapper(zipfit)
        if cov_kwds is None:
            cov_kwds = {}
        result._get_robustcov_results(cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)
        return result
    fit.__doc__ = DiscreteModel.fit.__doc__

    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if np.size(alpha) == 1 and alpha != 0:
            k_params = self.exog.shape[1]
            alpha = alpha * np.ones(k_params)
        alpha_p = alpha
        if start_params is None:
            offset = getattr(self, 'offset', 0) + getattr(self, 'exposure', 0)
            if np.size(offset) == 1 and offset == 0:
                offset = None
            model = self.model_main.__class__(self.endog, self.exog, offset=offset)
            start_params = model.fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=0, callback=callback, alpha=alpha_p, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs).params
        cntfit = super(CountModel, self).fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        if method in ['l1', 'l1_cvxopt_cp']:
            discretefit = self.result_class_reg(self, cntfit)
        else:
            raise TypeError('argument method == %s, which is not handled' % method)
        return self.result_class_reg_wrapper(discretefit)
    fit_regularized.__doc__ = DiscreteModel.fit_regularized.__doc__

    def hessian(self, params):
        """
        Generic Censored model Hessian matrix of the loglikelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        """
        return approx_hess(params, self.loglike)