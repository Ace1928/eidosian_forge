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
def _initialize_error_transition(self):
    if self.error_order == 0:
        self._initialize_error_transition_white_noise()
    else:
        k_endog = self.k_endog
        k_factors = self.k_factors
        _factor_order = self._factor_order
        _error_order = self._error_order
        _slice = np.s_['selection', _factor_order:_factor_order + k_endog, k_factors:k_factors + k_endog]
        self.ssm[_slice] = np.eye(k_endog)
        _slice = np.s_['transition', _factor_order + k_endog:_factor_order + _error_order, _factor_order:_factor_order + _error_order - k_endog]
        self.ssm[_slice] = np.eye(_error_order - k_endog)
        if self.error_var:
            self._initialize_error_transition_var()
        else:
            self._initialize_error_transition_individual()