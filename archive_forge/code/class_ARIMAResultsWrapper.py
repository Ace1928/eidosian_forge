from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.tools import diff
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.gls import gls as estimate_gls
from statsmodels.tsa.arima.specification import SARIMAXSpecification
class ARIMAResultsWrapper(sarimax.SARIMAXResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(sarimax.SARIMAXResultsWrapper._wrap_attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(sarimax.SARIMAXResultsWrapper._wrap_methods, _methods)