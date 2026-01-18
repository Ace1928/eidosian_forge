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
class TruncatedLFPoisson(TruncatedLFGeneric):
    __doc__ = '\n    Truncated Poisson model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, truncation=0, missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, truncation=truncation, missing=missing, **kwargs)
        self.model_main = Poisson(self.endog, self.exog, exposure=getattr(self, 'exposure', None), offset=getattr(self, 'offset', None))
        self.model_dist = truncatedpoisson
        self.result_class = TruncatedLFPoissonResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _predict_mom_trunc0(self, params, mu):
        """Predict mean and variance of zero-truncated distribution.

        experimental api, will likely be replaced by other methods

        Parameters
        ----------
        params : array_like
            The model parameters. This is only used to extract extra params
            like dispersion parameter.
        mu : array_like
            Array of mean predictions for main model.

        Returns
        -------
        Predicted conditional variance.
        """
        w = 1 - np.exp(-mu)
        m = mu / w
        var_ = m - (1 - w) * m ** 2
        return (m, var_)