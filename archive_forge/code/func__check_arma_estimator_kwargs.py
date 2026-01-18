import numpy as np
import warnings
from statsmodels.tools.tools import add_constant, Bunch
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
def _check_arma_estimator_kwargs(kwargs, method):
    if kwargs:
        raise ValueError(f'arma_estimator_kwargs not supported for method {method}')