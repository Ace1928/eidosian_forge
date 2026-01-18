from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
class ETSResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'fittedvalues': 'rows', 'level': 'rows', 'resid': 'rows', 'season': 'rows', 'slope': 'rows'}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'predict': 'dates', 'forecast': 'dates'}
    _wrap_methods = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_methods, _methods)