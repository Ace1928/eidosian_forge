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
def _initialize_factor_transition(self):
    order = self.factor_order * self.k_factors
    k_factors = self.k_factors
    self.parameters['factor_transition'] = self.factor_order * self.k_factors ** 2
    if self.k_factors > 0:
        if self.factor_order > 0:
            self.ssm['transition', k_factors:order, :order - k_factors] = np.eye(order - k_factors)
        self.ssm['selection', :k_factors, :k_factors] = np.eye(k_factors)
        self.ssm['state_cov', :k_factors, :k_factors] = np.eye(k_factors)
    self._idx_factor_transition = np.s_['transition', :k_factors, :order]