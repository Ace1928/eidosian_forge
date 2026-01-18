import numpy as np
from statsmodels.tools.tools import Bunch
from statsmodels.regression import linear_model
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

    Estimate AR parameters using Burg technique.

    Parameters
    ----------
    endog : array_like or SARIMAXSpecification
        Input time series array, assumed to be stationary.
    ar_order : int, optional
        Autoregressive order. Default is 0.
    demean : bool, optional
        Whether to estimate and remove the mean from the process prior to
        fitting the autoregressive coefficients.

    Returns
    -------
    parameters : SARIMAXParams object
        Contains the parameter estimates from the final iteration.
    other_results : Bunch
        Includes one component, `spec`, which is the `SARIMAXSpecification`
        instance corresponding to the input arguments.

    Notes
    -----
    The primary reference is [1]_, section 5.1.2.

    This procedure assumes that the series is stationary.

    This function is a light wrapper around `statsmodels.linear_model.burg`.

    References
    ----------
    .. [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    