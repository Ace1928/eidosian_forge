import numpy as np
def acovf_fft(x, demean=True):
    """autocovariance function with call to fftconvolve, biased

    Parameters
    ----------
    x : array_like
        timeseries, signal
    demean : bool
        If true, then demean time series

    Returns
    -------
    acovf : ndarray
        autocovariance for data, same length as x

    might work for nd in parallel with time along axis 0

    """
    from scipy import signal
    x = np.asarray(x)
    if demean:
        x = x - x.mean()
    signal.fftconvolve(x, x[::-1])[len(x) - 1:len(x) + 10] / x.shape[0]