import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender
def _initialize_error_transition_var(self):
    k_endog = self.k_endog
    _factor_order = self._factor_order
    _error_order = self._error_order
    self.parameters['error_transition'] = _error_order * k_endog
    self._idx_error_transition = np.s_['transition', _factor_order:_factor_order + k_endog, _factor_order:_factor_order + _error_order]