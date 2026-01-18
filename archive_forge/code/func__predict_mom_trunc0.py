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
    alpha = params[-1]
    p = self.model_main.parameterization
    prob_zero = (1 + alpha * mu ** (p - 1)) ** (-1 / alpha)
    w = 1 - prob_zero
    m = mu / w
    vm = mu * (1 + alpha * mu ** (p - 1))
    mnc2 = (mu ** 2 + vm) / w
    var_ = mnc2 - m ** 2
    return (m, var_)