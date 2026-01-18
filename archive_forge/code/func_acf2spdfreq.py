import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def acf2spdfreq(self, acovf, nfreq=100, w=None):
    """
        not really a method
        just for comparison, not efficient for large n or long acf

        this is also similarly use in tsa.stattools.periodogram with window
        """
    if w is None:
        w = np.linspace(0, np.pi, nfreq)[:, None]
    nac = len(acovf)
    hw = 0.5 / np.pi * (acovf[0] + 2 * (acovf[1:] * np.cos(w * np.arange(1, nac))).sum(1))
    return hw