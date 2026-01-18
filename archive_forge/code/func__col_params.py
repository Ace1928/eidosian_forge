from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def _col_params(result, float_format='%.4f', stars=True, include_r2=False):
    """Stack coefficients and standard errors in single column
    """
    res = summary_params(result)
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: float_format % x)
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    if stars:
        idx = res.iloc[:, 3] < 0.1
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < 0.05
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < 0.01
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
    res = res.iloc[:, :2]
    res = res.stack(**FUTURE_STACK)
    if include_r2:
        rsquared = getattr(result, 'rsquared', np.nan)
        rsquared_adj = getattr(result, 'rsquared_adj', np.nan)
        r2 = pd.Series({('R-squared', ''): rsquared, ('R-squared Adj.', ''): rsquared_adj})
        if r2.notnull().any():
            r2 = r2.apply(lambda x: float_format % x)
            res = pd.concat([res, r2], axis=0)
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res