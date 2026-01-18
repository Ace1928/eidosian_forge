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
class ZeroInflatedGeneralizedPoisson(GenericZeroInflated):
    __doc__ = '\n    Zero Inflated Generalized Poisson Model\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : ndarray\n        A reference to the endogenous response variable\n    exog : ndarray\n        A reference to the exogenous design.\n    exog_infl : ndarray\n        A reference to the zero-inflated exogenous design.\n    p : scalar\n        P denotes parametrizations for ZIGP regression.\n    ' % {'params': base._model_params_doc, 'extra_params': _doc_zi_params + 'p : float\n        dispersion power parameter for the GeneralizedPoisson model.  p=1 for\n        ZIGP-1 and p=2 for ZIGP-2. Default is p=2\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, exog_infl=None, offset=None, exposure=None, inflation='logit', p=2, missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, inflation=inflation, exog_infl=exog_infl, exposure=exposure, missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog, offset=offset, exposure=exposure, p=p)
        self.distribution = zigenpoisson
        self.k_exog += 1
        self.k_extra += 1
        self.exog_names.append('alpha')
        self.result_class = ZeroInflatedGeneralizedPoissonResults
        self.result_class_wrapper = ZeroInflatedGeneralizedPoissonResultsWrapper
        self.result_class_reg = L1ZeroInflatedGeneralizedPoissonResults
        self.result_class_reg_wrapper = L1ZeroInflatedGeneralizedPoissonResultsWrapper

    def _get_init_kwds(self):
        kwds = super()._get_init_kwds()
        kwds['p'] = self.model_main.parameterization + 1
        return kwds

    def _predict_prob(self, params, exog, exog_infl, exposure, offset, y_values=None):
        params_infl = params[:self.k_inflate]
        params_main = params[self.k_inflate:]
        p = self.model_main.parameterization + 1
        if y_values is None:
            y_values = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
        if len(exog_infl.shape) < 2:
            transform = True
            w = np.atleast_2d(self.model_infl.predict(params_infl, exog_infl))[:, None]
        else:
            transform = False
            w = self.model_infl.predict(params_infl, exog_infl)[:, None]
        w[w == 1.0] = np.nextafter(1, 0)
        mu = self.model_main.predict(params_main, exog, exposure=exposure, offset=offset)[:, None]
        result = self.distribution.pmf(y_values, mu, params_main[-1], p, w)
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
        alpha = params[-1]
        w = prob_infl
        p = self.model_main.parameterization
        var_ = (1 - w) * mu * ((1 + alpha * mu ** p) ** 2 + w * mu)
        return var_

    def _get_start_params(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            start_params = ZeroInflatedPoisson(self.endog, self.exog, exog_infl=self.exog_infl).fit(disp=0).params
        start_params = np.append(start_params, 0.1)
        return start_params

    @Appender(ZeroInflatedPoisson.get_distribution.__doc__)
    def get_distribution(self, params, exog=None, exog_infl=None, exposure=None, offset=None):
        p = self.model_main.parameterization + 1
        mu = self.predict(params, exog=exog, exog_infl=exog_infl, exposure=exposure, offset=offset, which='mean-main')
        w = self.predict(params, exog=exog, exog_infl=exog_infl, exposure=exposure, offset=offset, which='prob-main')
        distr = self.distribution(mu, params[-1], p, 1 - w)
        return distr