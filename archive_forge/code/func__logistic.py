import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _logistic(x):
    """
    Note that this is not a vectorized function
    """
    x = np.array(x)
    if x.ndim == 0:
        y = np.reshape(x, (1, 1, 1))
    elif x.ndim == 1:
        y = np.reshape(x, (len(x), 1, 1))
    elif x.ndim == 2:
        y = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    elif x.ndim == 3:
        y = x
    else:
        raise NotImplementedError
    tmp = np.c_[np.zeros((y.shape[-1], y.shape[1], 1)), y.T].T
    evaluated = np.reshape(np.exp(y - logsumexp(tmp, axis=0)), x.shape)
    return evaluated