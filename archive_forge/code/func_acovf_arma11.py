import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def acovf_arma11(ar, ma):
    a = -ar[1]
    b = ma[1]
    rho = [(1.0 + b ** 2 + 2 * a * b) / (1.0 - a ** 2)]
    rho.append((1 + a * b) * (a + b) / (1.0 - a ** 2))
    for _ in range(8):
        last = rho[-1]
        rho.append(a * last)
    return np.array(rho)