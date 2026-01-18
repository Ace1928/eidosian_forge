import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
class KimSmootherResults(HamiltonFilterResults):
    """
    Results from applying the Kim smoother to a Markov switching model.

    Parameters
    ----------
    model : MarkovSwitchingModel
        The model object.
    result : dict
        A dictionary containing two keys: 'smoothd_joint_probabilities' and
        'smoothed_marginal_probabilities'.

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    """

    def __init__(self, model, result):
        super().__init__(model, result)
        attributes = ['smoothed_joint_probabilities', 'smoothed_marginal_probabilities']
        for name in attributes:
            setattr(self, name, getattr(result, name))