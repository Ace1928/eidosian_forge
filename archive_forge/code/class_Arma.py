class Arma:
    """
    Removed in 0.14. Use SARIMAX, ARIMA or AutoReg.

    See Also
    --------
    statsmodels.tsa.statespace.sarimax.SARIMAX
    statsmodels.tsa.arima.model.ARIMA
    statsmodels.tsa.ar_model.AutoReg
    """

    def __init__(self, endog, exog=None):
        raise NotImplementedError('ARMA has been removed. Use SARIMAX, ARIMA or AutoReg')