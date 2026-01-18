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
def _initialize_error_cov_diagonal(self, scalar=False):
    self.parameters['error_cov'] = 1 if scalar else self.k_endog
    k_endog = self.k_endog
    k_factors = self.k_factors
    idx = np.diag_indices(k_endog)
    if self.error_order > 0:
        matrix = 'state_cov'
        idx = (idx[0] + k_factors, idx[1] + k_factors)
    else:
        matrix = 'obs_cov'
    self._idx_error_cov = (matrix,) + idx