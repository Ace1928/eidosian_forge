from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def _set_exog_names(self, vals):
    if not isinstance(vals, list):
        vals = [vals]
    self.data.xnames = vals