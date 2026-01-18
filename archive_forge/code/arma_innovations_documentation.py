import numpy as np
from statsmodels.tsa import arima_process
from statsmodels.tsa.statespace.tools import prefix_dtype_map
from statsmodels.tools.numdiff import _get_epsilon, approx_fprime_cs
from scipy.linalg.blas import find_best_blas_type
from . import _arma_innovations

    Compute the score (gradient) per observation.

    Parameters
    ----------
    endog : ndarray
        The observed time-series process.
    ar_params : ndarray, optional
        Autoregressive coefficients, not including the zero lag.
    ma_params : ndarray, optional
        Moving average coefficients, not including the zero lag, where the sign
        convention assumes the coefficients are part of the lag polynomial on
        the right-hand-side of the ARMA definition (i.e. they have the same
        sign from the usual econometrics convention in which the coefficients
        are on the right-hand-side of the ARMA definition).
    sigma2 : ndarray, optional
        The ARMA innovation variance. Default is 1.
    prefix : str, optional
        The BLAS prefix associated with the datatype. Default is to find the
        best datatype based on given input. This argument is typically only
        used internally.

    Returns
    -------
    ndarray
        Score per observation, evaluated at the given parameters.

    Notes
    -----
    This is a numerical approximation, calculated using first-order complex
    step differentiation on the `arma_loglike` method.
    