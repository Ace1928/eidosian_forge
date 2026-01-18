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
class CountModel(DiscreteModel):

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none', check_rank=True, **kwargs):
        self._check_kwargs(kwargs)
        super().__init__(endog, exog, check_rank, missing=missing, offset=offset, exposure=exposure, **kwargs)
        if exposure is not None:
            self.exposure = np.asarray(self.exposure)
            self.exposure = np.log(self.exposure)
        if offset is not None:
            self.offset = np.asarray(self.offset)
        self._check_inputs(self.offset, self.exposure, self.endog)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')
        dt = np.promote_types(self.endog.dtype, np.float64)
        self.endog = np.asarray(self.endog, dt)
        dt = np.promote_types(self.exog.dtype, np.float64)
        self.exog = np.asarray(self.exog, dt)

    def _check_inputs(self, offset, exposure, endog):
        if offset is not None and offset.shape[0] != endog.shape[0]:
            raise ValueError('offset is not the same length as endog')
        if exposure is not None and exposure.shape[0] != endog.shape[0]:
            raise ValueError('exposure is not the same length as endog')

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def _get_predict_arrays(self, exog=None, offset=None, exposure=None):
        if exposure is not None:
            exposure = np.log(np.asarray(exposure))
        if offset is not None:
            offset = np.asarray(offset)
        if exog is None:
            exog = self.exog
            if exposure is None:
                exposure = getattr(self, 'exposure', 0)
            if offset is None:
                offset = getattr(self, 'offset', 0)
        else:
            exog = np.asarray(exog)
            if exposure is None:
                exposure = 0
            if offset is None:
                offset = 0
        return (exog, offset, exposure)

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', linear=None):
        """
        Predict response variable of a count model given exogenous variables

        Parameters
        ----------
        params : array_like
            Model parameters
        exog : array_like, optional
            Design / exogenous data. Is exog is None, model exog is used.
        exposure : array_like, optional
            Log(exposure) is added to the linear prediction with
            coefficient equal to 1. If exposure is not provided and exog
            is None, uses the model's exposure if present.  If not, uses
            0 as the default value.
        offset : array_like, optional
            Offset is added to the linear prediction with coefficient
            equal to 1. If offset is not provided and exog
            is None, uses the model's offset if present.  If not, uses
            0 as the default value.
        which : 'mean', 'linear', 'var', 'prob' (optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' returns the conditional expectation of endog E(y | x),
              i.e. exp of linear predictor.
            - 'linear' returns the linear predictor of the mean function.
            - 'var' variance of endog implied by the likelihood model
            - 'prob' predicted probabilities for counts.

            .. versionadded: 0.14

               ``which`` replaces and extends the deprecated ``linear``
               argument.

        linear : bool
            If True, returns the linear predicted values.  If False or None,
            then the statistic specified by ``which`` will be returned.

            .. deprecated: 0.14

               The ``linear` keyword is deprecated and will be removed,
               use ``which`` keyword instead.


        Notes
        -----
        If exposure is specified, then it will be logged by the method.
        The user does not need to log it first.
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
        if exposure is None and exog is None and hasattr(self, 'exposure'):
            exposure = self.exposure
        elif exposure is None:
            exposure = 0.0
        else:
            exposure = np.log(exposure)
        if exog is None:
            exog = self.exog
        fitted = np.dot(exog, params[:exog.shape[1]])
        linpred = fitted + exposure + offset
        if which == 'mean':
            return np.exp(linpred)
        elif which.startswith('lin'):
            return linpred
        else:
            raise ValueError('keyword which has to be "mean" and "linear"')

    def _derivative_predict(self, params, exog=None, transform='dydx'):
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
        dF = self.predict(params, exog)[:, None] * exog
        if 'ey' in transform:
            dF /= self.predict(params, exog)[:, None]
        return dF

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        """
        For computing marginal effects. These are the marginal effects
        d F(XB) / dX
        For the Poisson model F(XB) is the predicted counts rather than
        the probabilities.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        if exog is None:
            exog = self.exog
        k_extra = getattr(self, 'k_extra', 0)
        params_exog = params if k_extra == 0 else params[:-k_extra]
        margeff = self.predict(params, exog)[:, None] * params_exog[None, :]
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
        from statsmodels.genmod.families import links
        link = links.Log()
        lin_pred = self.predict(params, which='linear')
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        if self.k_extra > 0:
            dmat_extra = np.zeros((dmat.shape[0], self.k_extra))
            dmat = np.column_stack((dmat, dmat_extra))
        return dmat

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        cntfit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        discretefit = CountResults(self, cntfit)
        return CountResultsWrapper(discretefit)

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        _validate_l1_method(method)
        cntfit = super().fit_regularized(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        discretefit = L1CountResults(self, cntfit)
        return L1CountResultsWrapper(discretefit)