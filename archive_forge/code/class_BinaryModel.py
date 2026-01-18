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
class BinaryModel(DiscreteModel):
    _continuous_ok = False

    def __init__(self, endog, exog, offset=None, check_rank=True, **kwargs):
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, offset=offset, check_rank=check_rank, **kwargs)
        if not issubclass(self.__class__, MultinomialModel):
            if not np.all((self.endog >= 0) & (self.endog <= 1)):
                raise ValueError('endog must be in the unit interval.')
        if offset is None:
            delattr(self, 'offset')
            if not self._continuous_ok and np.any(self.endog != np.round(self.endog)):
                raise ValueError('endog must be binary, either 0 or 1')

    def predict(self, params, exog=None, which='mean', linear=None, offset=None):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array_like
            Fitted parameters of the model.
        exog : array_like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used.
        which : {'mean', 'linear', 'var', 'prob'}, optional
            Statistic to predict. Default is 'mean'.

            - 'mean' returns the conditional expectation of endog E(y | x),
              i.e. exp of linear predictor.
            - 'linear' returns the linear predictor of the mean function.
            - 'var' returns the estimated variance of endog implied by the
              model.

            .. versionadded: 0.14

               ``which`` replaces and extends the deprecated ``linear``
               argument.

        linear : bool
            If True, returns the linear predicted values.  If False or None,
            then the statistic specified by ``which`` will be returned.

            .. deprecated: 0.14

               The ``linear` keyword is deprecated and will be removed,
               use ``which`` keyword instead.

        Returns
        -------
        array
            Fitted values at exog.
        """
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.0
        if exog is None:
            exog = self.exog
        linpred = np.dot(exog, params) + offset
        if which == 'mean':
            return self.cdf(linpred)
        elif which == 'linear':
            return linpred
        if which == 'var':
            mu = self.cdf(linpred)
            var_ = mu * (1 - mu)
            return var_
        else:
            raise ValueError('Only `which` is "mean", "linear" or "var" are available.')

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        _validate_l1_method(method)
        bnryfit = super().fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1BinaryResults(self, bnryfit)
        return L1BinaryResultsWrapper(discretefit)

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        res = fit_constrained_wrap(self, constraints, start_params=None, **fit_kwds)
        return res
    fit_constrained.__doc__ = fit_constrained_wrap.__doc__

    def _derivative_predict(self, params, exog=None, transform='dydx', offset=None):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predict.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        linpred = self.predict(params, exog, offset=offset, which='linear')
        dF = self.pdf(linpred)[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog, offset=offset)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None, offset=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        if exog is None:
            exog = self.exog
        linpred = self.predict(params, exog, offset=offset, which='linear')
        margeff = np.dot(self.pdf(linpred)[:, None], params[None, :])
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
        return self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        link = self.link
        lin_pred = self.predict(params, which='linear')
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return dmat

    def get_distribution(self, params, exog=None, offset=None):
        """Get frozen instance of distribution based on predicted parameters.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        offset : ndarray, optional
            Offset is added to the linear predictor of the mean function with
            coefficient equal to 1.
            Default is zero if exog is not None, and the model offset if exog
            is None.
        exposure : ndarray, optional
            Log(exposure) is added to the linear predictor  of the mean
            function with coefficient equal to 1. If exposure is specified,
            then it will be logged by the method. The user does not need to
            log it first.
            Default is one if exog is is not None, and it is the model exposure
            if exog is None.

        Returns
        -------
        Instance of frozen scipy distribution.
        """
        mu = self.predict(params, exog=exog, offset=offset)
        distr = stats.bernoulli(mu)
        return distr