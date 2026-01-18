class ARMA:
    """
    ARMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(ARIMA_DEPRECATION_ERROR)