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
def _wrap_results(self, params, result, return_raw, cov_type=None, cov_kwds=None, results_class=None, wrapper_class=None):
    if not return_raw:
        result_kwargs = {}
        if cov_type is not None:
            result_kwargs['cov_type'] = cov_type
        if cov_kwds is not None:
            result_kwargs['cov_kwds'] = cov_kwds
        if results_class is None:
            results_class = self._res_classes['fit'][0]
        if wrapper_class is None:
            wrapper_class = self._res_classes['fit'][1]
        res = results_class(self, params, result, **result_kwargs)
        result = wrapper_class(res)
    return result