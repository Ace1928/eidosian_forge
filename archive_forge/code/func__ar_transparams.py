from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def _ar_transparams(params):
    """
    Transforms params to induce stationarity/invertability.

    Parameters
    ----------
    params : array_like
        The AR coefficients

    Reference
    ---------
    Jones(1980)
    """
    newparams = np.tanh(params / 2)
    tmp = np.tanh(params / 2)
    for j in range(1, len(params)):
        a = newparams[j]
        for kiter in range(j):
            tmp[kiter] -= a * newparams[j - kiter - 1]
        newparams[:j] = tmp[:j]
    return newparams