from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
class ResultMixin:

    @cache_readonly
    def df_modelwc(self):
        """Model WC"""
        k_extra = getattr(self.model, 'k_extra', 0)
        if hasattr(self, 'df_model'):
            if hasattr(self, 'k_constant'):
                hasconst = self.k_constant
            elif hasattr(self, 'hasconst'):
                hasconst = self.hasconst
            else:
                hasconst = 1
            return self.df_model + hasconst + k_extra
        else:
            return self.params.size

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        return -2 * self.llf + 2 * self.df_modelwc

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        return -2 * self.llf + np.log(self.nobs) * self.df_modelwc

    @cache_readonly
    def score_obsv(self):
        """cached Jacobian of log-likelihood
        """
        return self.model.score_obs(self.params)

    @cache_readonly
    def hessv(self):
        """cached Hessian of log-likelihood
        """
        return self.model.hessian(self.params)

    @cache_readonly
    def covjac(self):
        """
        covariance of parameters based on outer product of jacobian of
        log-likelihood
        """
        jacv = self.score_obsv
        return np.linalg.inv(np.dot(jacv.T, jacv))

    @cache_readonly
    def covjhj(self):
        """covariance of parameters based on HJJH

        dot product of Hessian, Jacobian, Jacobian, Hessian of likelihood

        name should be covhjh
        """
        jacv = self.score_obsv
        hessv = self.hessv
        hessinv = np.linalg.inv(hessv)
        return np.dot(hessinv, np.dot(np.dot(jacv.T, jacv), hessinv))

    @cache_readonly
    def bsejhj(self):
        """standard deviation of parameter estimates based on covHJH
        """
        return np.sqrt(np.diag(self.covjhj))

    @cache_readonly
    def bsejac(self):
        """standard deviation of parameter estimates based on covjac
        """
        return np.sqrt(np.diag(self.covjac))

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        """simple bootstrap to get mean and variance of estimator

        see notes

        Parameters
        ----------
        nrep : int
            number of bootstrap replications
        method : str
            optimization method to use
        disp : bool
            If true, then optimization prints results
        store : bool
            If true, then parameter estimates for all bootstrap iterations
            are attached in self.bootstrap_results

        Returns
        -------
        mean : ndarray
            mean of parameter estimates over bootstrap replications
        std : ndarray
            standard deviation of parameter estimates over bootstrap
            replications

        Notes
        -----
        This was mainly written to compare estimators of the standard errors of
        the parameter estimates.  It uses independent random sampling from the
        original endog and exog, and therefore is only correct if observations
        are independently distributed.

        This will be moved to apply only to models with independently
        distributed observations.
        """
        results = []
        hascloneattr = True if hasattr(self.model, 'cloneattr') else False
        for i in range(nrep):
            rvsind = np.random.randint(self.nobs, size=self.nobs)
            if self.exog is not None:
                exog_resamp = self.exog[rvsind, :]
            else:
                exog_resamp = None
            init_kwds = self.model._get_init_kwds()
            fitmod = self.model.__class__(self.endog[rvsind], exog=exog_resamp, **init_kwds)
            if hascloneattr:
                for attr in self.model.cloneattr:
                    setattr(fitmod, attr, getattr(self.model, attr))
            fitres = fitmod.fit(method=method, disp=disp)
            results.append(fitres.params)
        results = np.array(results)
        if store:
            self.bootstrap_results = results
        return (results.mean(0), results.std(0), results)

    def get_nlfun(self, fun):
        """
        get_nlfun

        This is not Implemented
        """
        raise NotImplementedError