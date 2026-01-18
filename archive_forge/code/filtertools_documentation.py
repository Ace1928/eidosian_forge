import numpy as np
import scipy.fftpack as fft
from scipy import signal
from statsmodels.tools.validation import array_like, PandasWrapper

    Filter multiple time series into a single time series.

    Uses a convolution to merge inputs, and then lfilter to produce output.

    Parameters
    ----------
    ar : array_like
        The coefficients of autoregressive lag polynomial including lag zero,
        ar(L) in the expression ar(L)y_t.
    ma : array_like, same ndim as x, currently 2d
        The coefficient of the moving average lag polynomial, ma(L) in
        ma(L)x_t.
    x : array_like
        The 2-d input data series, time in rows, variables in columns.
    useic : bool
        Flag indicating whether to use initial conditions.

    Returns
    -------
    y : ndarray
        The filtered output series.
    inp : ndarray, 1d
        The combined input series.

    Notes
    -----
    currently for 2d inputs only, no choice of axis
    Use of signal.lfilter requires that ar lag polynomial contains
    floating point numbers
    does not cut off invalid starting and final values

    miso_lfilter find array y such that:

            ar(L)y_t = ma(L)x_t

    with shapes y (nobs,), x (nobs, nvars), ar (narlags,), and
    ma (narlags, nvars).
    