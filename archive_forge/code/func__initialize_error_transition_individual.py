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
def _initialize_error_transition_individual(self):
    k_endog = self.k_endog
    _error_order = self._error_order
    self.parameters['error_transition'] = _error_order
    idx = np.tile(np.diag_indices(k_endog), self.error_order)
    row_shift = self._factor_order
    col_inc = self._factor_order + np.repeat([i * k_endog for i in range(self.error_order)], k_endog)
    idx[0] += row_shift
    idx[1] += col_inc
    idx_diag = idx.copy()
    idx_diag[0] -= row_shift
    idx_diag[1] -= self._factor_order
    idx_diag = idx_diag[:, np.lexsort((idx_diag[1], idx_diag[0]))]
    self._idx_error_diag = (idx_diag[0], idx_diag[1])
    idx = idx[:, np.lexsort((idx[1], idx[0]))]
    self._idx_error_transition = np.s_['transition', idx[0], idx[1]]