import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def acovf_ma2(ma):
    b1 = -ma[1]
    b2 = -ma[2]
    rho = np.zeros(10)
    rho[0] = 1 + b1 ** 2 + b2 ** 2
    rho[1] = -b1 + b1 * b2
    rho[2] = -b2
    return rho