from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def _construct_variable_names(self):
    """Construct model variables names"""
    endog = self.data.orig_endog
    if isinstance(endog, pd.Series):
        y_base = endog.name or 'y'
    elif isinstance(endog, pd.DataFrame):
        y_base = endog.squeeze().name or 'y'
    else:
        y_base = 'y'
    y_name = f'D.{y_base}'
    x_names = list(self._deterministic_reg.columns)
    x_names.append(f'{y_base}.L1')
    orig_exog = self.data.orig_exog
    exog_pandas = isinstance(orig_exog, pd.DataFrame)
    dexog_names = []
    for key, val in self._order.items():
        if val is not None:
            if exog_pandas:
                x_name = f'{key}.L1'
            else:
                x_name = f'x{key}.L1'
            x_names.append(x_name)
            lag_base = x_name[:-1]
            for lag in val[:-1]:
                dexog_names.append(f'D.{lag_base}{lag}')
    y_lags = max(self._lags) if self._lags else 0
    dendog_names = [f'{y_name}.L{lag}' for lag in range(1, y_lags)]
    x_names.extend(dendog_names)
    x_names.extend(dexog_names)
    x_names.extend(self._fixed_names)
    return (y_name, x_names)