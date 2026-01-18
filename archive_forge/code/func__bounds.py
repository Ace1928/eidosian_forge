import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
def _bounds(self, params):
    """Integration bounds for the observation specific interval.

        This defines the lower and upper bounds for the intervals of the
        choices of all observations.

        The bounds for observation are given by

            a_{k_i-1} - linpred_i, a_k_i - linpred_i

        where
        - k_i is the choice in observation i.
        - a_{k_i-1} and a_k_i are thresholds (cutoffs) for choice k_i
        - linpred_i is the linear prediction for observation i

        Parameters
        ----------
        params : ndarray
            Parameters for the model, (exog_coef, transformed_thresholds)

        Return
        ------
        low : ndarray
            Lower bounds for choice intervals of each observation,
            1-dim with length nobs
        upp : ndarray
            Upper bounds for choice intervals of each observation,
            1-dim with length nobs.

        """
    thresh = self.transform_threshold_params(params)
    thresh_i_low = thresh[self.endog]
    thresh_i_upp = thresh[self.endog + 1]
    xb = self._linpred(params)
    low = thresh_i_low - xb
    upp = thresh_i_upp - xb
    return (low, upp)