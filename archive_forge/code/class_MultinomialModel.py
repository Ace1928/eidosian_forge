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
class MultinomialModel(BinaryModel):

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        if data_tools._is_using_ndarray_type(endog, None):
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'
        elif data_tools._is_using_pandas(endog, None):
            endog_dummies, ynames, yname = _pandas_to_dummies(endog)
        else:
            endog = np.asarray(endog)
            endog_dummies, ynames = _numpy_to_dummies(endog)
            yname = 'y'
        if not isinstance(ynames, dict):
            ynames = dict(zip(range(endog_dummies.shape[1]), ynames))
        self._ynames_map = ynames
        data = handle_data(endog_dummies, exog, missing, hasconst, **kwargs)
        data.ynames = yname
        data.orig_endog = endog
        self.wendog = data.endog
        for key in kwargs:
            if key in ['design_info', 'formula']:
                continue
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:
                pass
        return data

    def initialize(self):
        """
        Preprocesses the data for MNLogit.
        """
        super().initialize()
        self.endog = self.endog.argmax(1)
        self.J = self.wendog.shape[1]
        self.K = self.exog.shape[1]
        self.df_model *= self.J - 1
        self.df_resid = self.exog.shape[0] - self.df_model - (self.J - 1)

    def predict(self, params, exog=None, which='mean', linear=None):
        """
        Predict response variable of a model given exogenous variables.

        Parameters
        ----------
        params : array_like
            2d array of fitted parameters of the model. Should be in the
            order returned from the model.
        exog : array_like
            1d or 2d array of exogenous values.  If not supplied, the
            whole exog attribute of the model is used. If a 1d array is given
            it assumed to be 1 row of exogenous variables. If you only have
            one regressor and would like to do prediction, you must provide
            a 2d array with shape[1] == 1.
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

        Notes
        -----
        Column 0 is the base case, the rest conform to the rows of params
        shifted up one for the base case.
        """
        if linear is not None:
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.warn(msg, FutureWarning)
            if linear is True:
                which = 'linear'
        if exog is None:
            exog = self.exog
        if exog.ndim == 1:
            exog = exog[None]
        pred = super().predict(params, exog, which=which)
        if which == 'linear':
            pred = np.column_stack((np.zeros(len(exog)), pred))
        return pred

    @Appender(DiscreteModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        if start_params is None:
            start_params = np.zeros(self.K * (self.J - 1))
        else:
            start_params = np.asarray(start_params)
        if callback is None:
            callback = lambda x, *args: None
        mnfit = base.LikelihoodModel.fit(self, start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = MultinomialResults(self, mnfit)
        return MultinomialResultsWrapper(mnfit)

    @Appender(DiscreteModel.fit_regularized.__doc__)
    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=1, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, **kwargs):
        if start_params is None:
            start_params = np.zeros(self.K * (self.J - 1))
        else:
            start_params = np.asarray(start_params)
        mnfit = DiscreteModel.fit_regularized(self, start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, alpha=alpha, trim_mode=trim_mode, auto_trim_tol=auto_trim_tol, size_trim_tol=size_trim_tol, qc_tol=qc_tol, **kwargs)
        mnfit.params = mnfit.params.reshape(self.K, -1, order='F')
        mnfit = L1MultinomialResults(self, mnfit)
        return L1MultinomialResultsWrapper(mnfit)

    def _derivative_predict(self, params, exog=None, transform='dydx'):
        """
        For computing marginal effects standard errors.

        This is used only in the case of discrete and count regressors to
        get the variance-covariance of the marginal effects. It returns
        [d F / d params] where F is the predicted probabilities for each
        choice. dFdparams is of shape nobs x (J*K) x (J-1)*K.
        The zero derivatives for the base category are not included.

        Transform can be 'dydx' or 'eydx'. Checking is done in margeff
        computations for appropriate transform.
        """
        if exog is None:
            exog = self.exog
        if params.ndim == 1:
            params = params.reshape(self.K, self.J - 1, order='F')
        eXB = np.exp(np.dot(exog, params))
        sum_eXB = (1 + eXB.sum(1))[:, None]
        J = int(self.J)
        K = int(self.K)
        repeat_eXB = np.repeat(eXB, J, axis=1)
        X = np.tile(exog, J - 1)
        F0 = -repeat_eXB * X / sum_eXB ** 2
        F1 = eXB.T[:, :, None] * X * (sum_eXB - repeat_eXB) / sum_eXB ** 2
        F1 = F1.transpose((1, 0, 2))
        other_idx = ~np.kron(np.eye(J - 1), np.ones(K)).astype(bool)
        F1[:, other_idx] = (-eXB.T[:, :, None] * X * repeat_eXB / sum_eXB ** 2).transpose((1, 0, 2))[:, other_idx]
        dFdX = np.concatenate((F0[:, None, :], F1), axis=1)
        if 'ey' in transform:
            dFdX /= self.predict(params, exog)[:, :, None]
        return dFdX

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.

        For Multinomial models the marginal effects are

        P[j] * (params[j] - sum_k P[k]*params[k])

        It is returned unshaped, so that each row contains each of the J
        equations. This makes it easier to take derivatives of this for
        standard errors. If you want average marginal effects you can do
        margeff.reshape(nobs, K, J, order='F).mean(0) and the marginal effects
        for choice J are in column J
        """
        J = int(self.J)
        K = int(self.K)
        if exog is None:
            exog = self.exog
        if params.ndim == 1:
            params = params.reshape(K, J - 1, order='F')
        zeroparams = np.c_[np.zeros(K), params]
        cdf = self.cdf(np.dot(exog, params))
        iterm = np.array([cdf[:, [i]] * zeroparams[:, i] for i in range(int(J))]).sum(0)
        margeff = np.array([cdf[:, [j]] * (zeroparams[:, j] - iterm) for j in range(J)])
        margeff = np.transpose(margeff, (1, 2, 0))
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None, :]
        margeff = self._derivative_exog_helper(margeff, params, exog, dummy_idx, count_idx, transform)
        return margeff.reshape(len(exog), -1, order='F')

    def get_distribution(self, params, exog=None, offset=None):
        """get frozen instance of distribution
        """
        raise NotImplementedError