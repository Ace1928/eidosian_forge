from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def _validate_out_of_sample_exog(self, exog, out_of_sample):
    """
        Validate given `exog` as satisfactory for out-of-sample operations

        Parameters
        ----------
        exog : array_like or None
            New observations of exogenous regressors, if applicable.
        out_of_sample : int
            Number of new observations required.

        Returns
        -------
        exog : array or None
            A numpy array of shape (out_of_sample, k_exog) if the model
            contains an `exog` component, or None if it does not.
        """
    k_exog = getattr(self, 'k_exog', 0)
    if out_of_sample and k_exog > 0:
        if exog is None:
            raise ValueError('Out-of-sample operations in a model with a regression component require additional exogenous values via the `exog` argument.')
        exog = np.array(exog)
        required_exog_shape = (out_of_sample, self.k_exog)
        try:
            exog = exog.reshape(required_exog_shape)
        except ValueError:
            raise ValueError('Provided exogenous values are not of the appropriate shape. Required %s, got %s.' % (str(required_exog_shape), str(exog.shape)))
    elif k_exog > 0 and exog is not None:
        exog = None
        warnings.warn('Exogenous array provided, but additional data is not required. `exog` argument ignored.', ValueWarning)
    return exog