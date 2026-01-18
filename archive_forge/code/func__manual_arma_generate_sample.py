from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
def _manual_arma_generate_sample(ar, ma, eta):
    T = len(eta)
    ar = ar[::-1]
    ma = ma[::-1]
    p, q = (len(ar), len(ma))
    rep2 = [0] * max(p, q)
    for t in range(T):
        yt = eta[t]
        if p:
            yt += np.dot(rep2[-p:], ar)
        if q:
            yt += np.dot([0] * (q - t) + list(eta[max(0, t - q):t]), ma)
        rep2.append(yt)
    return np.array(rep2[max(p, q):])