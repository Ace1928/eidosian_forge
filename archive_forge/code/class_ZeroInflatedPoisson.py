import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.discrete.discrete_model import (DiscreteModel, CountModel,
from statsmodels.distributions import zipoisson, zigenpoisson, zinegbin
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.compat.pandas import Appender
class ZeroInflatedPoisson(GenericZeroInflated):
    __doc__ = '\n    Poisson Zero Inflated Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, inflation=inflation, exog_infl=exog_infl, exposure=exposure, missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, offset=offset, exposure=exposure)
        self.distribution = zipoisson
        self.result_class = ZeroInflatedPoissonResults
        self.result_class_wrapper = ZeroInflatedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedPoissonResultsWrapper

    def _hessian_main(self, params):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]
        y = self.endog
        w = self.model_infl.predict(params_infl)
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        score = self.score(params)
        zero_idx = np.nonzero(y == 0)[0]
        nonzero_idx = np.nonzero(y)[0]
        mu = self.model_main.predict(params_main)
        hess_arr = np.zeros((self.k_exog, self.k_exog))
        coeff = 1 + w[zero_idx] * (np.exp(mu[zero_idx]) - 1)
        for i in range(self.k_exog):
            for j in range(i, -1, -1):
                hess_arr[i, j] = (self.exog[zero_idx, i] * self.exog[zero_idx, j] * mu[zero_idx] * (w[zero_idx] - 1) * (1 / coeff - w[zero_idx] * mu[zero_idx] * np.exp(mu[zero_idx]) / coeff ** 2)).sum() - (mu[nonzero_idx] * self.exog[nonzero_idx, i] * self.exog[nonzero_idx, j]).sum()
        return hess_arr

    def _predict_prob(self, params, exog, exog_infl, exposure, offset, y_values=None):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]
        if y_values is None:
            y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(self.model_infl.predict(params_infl, exog_infl))[:, None]
        else:
            transform = False
            w = self.model_infl.predict(params_infl, exog_infl)[:, None]
        w = np.clip(w, np.finfo(float).eps, 1 - np.finfo(float).eps)
        mu = self.model_main.predict(params_main, exog, offset=offset)[:, None]
        result = self.distribution.pmf(y_values, mu, w)
        return result[0] if transform else result

    def _predict_var(self, params, mu, prob_infl):
        """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.
        prob_inlf : array_like
            Array of predicted probabilities of zero-inflation `w`.

        Returns
        -------
        Predicted conditional variance.
        """
        w = prob_infl
        var_ = (1 - w) * mu * (1 + w * mu)
        return var_

    def _get_start_params(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            start_params = self.model_main.fit(disp=0, method='nm').params
        start_params = np.append(np.ones(self.k_inflate) * 0.1, start_params)
        return start_params

    def get_distribution(self, params, exog=None, exog_infl=None, exposure=None, offset=None):
        """Get frozen instance of distribution based on predicted parameters.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : ndarray, optional
            Explanatory variables for the main count model.
            If ``exog`` is None, then the data from the model will be used.
        exog_infl : ndarray, optional
            Explanatory variables for the zero-inflation model.
            ``exog_infl`` has to be provided if ``exog`` was provided unless
            ``exog_infl`` in the model is only a constant.
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
        Instance of frozen scipy distribution subclass.
        """
        mu = self.predict(params, exog=exog, exog_infl=exog_infl, exposure=exposure, offset=offset, which='mean-main')
        w = self.predict(params, exog=exog, exog_infl=exog_infl, exposure=exposure, offset=offset, which='prob-main')
        distr = self.distribution(mu, 1 - w)
        return distr