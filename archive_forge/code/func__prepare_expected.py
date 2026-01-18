from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
@staticmethod
def _prepare_expected(data, lags, trim='front'):
    t, k = data.shape
    expected = np.zeros((t + lags, (lags + 1) * k))
    for col in range(k):
        for i in range(lags + 1):
            if i < lags:
                expected[i:-lags + i, (lags + 1) * col + i] = data[:, col]
            else:
                expected[i:, (lags + 1) * col + i] = data[:, col]
    if trim == 'front':
        expected = expected[:-lags]
    return expected