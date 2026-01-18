import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import (
from scipy.stats.distributions import rv_frozen
from statsmodels.base.data import PandasData
from statsmodels.base.model import Results
from statsmodels.base.wrapper import (
class HoltWintersResultsWrapper(ResultsWrapper):
    _attrs = {'fittedvalues': 'rows', 'level': 'rows', 'resid': 'rows', 'season': 'rows', 'trend': 'rows', 'slope': 'rows'}
    _wrap_attrs = union_dicts(ResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates', 'forecast': 'dates'}
    _wrap_methods = union_dicts(ResultsWrapper._wrap_methods, _methods)