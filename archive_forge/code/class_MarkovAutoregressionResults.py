import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
class MarkovAutoregressionResults(markov_regression.MarkovRegressionResults):
    """
    Class to hold results from fitting a Markov switching autoregression model

    Parameters
    ----------
    model : MarkovAutoregression instance
        The fitted model instance
    params : ndarray
        Fitted parameters
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    cov_type : str
        The type of covariance matrix estimator to use. Can be one of 'approx',
        'opg', 'robust', or 'none'.

    Attributes
    ----------
    model : Model instance
        A reference to the model that was fit.
    filter_results : HamiltonFilterResults or KimSmootherResults instance
        The underlying filter and, optionally, smoother output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    """
    pass