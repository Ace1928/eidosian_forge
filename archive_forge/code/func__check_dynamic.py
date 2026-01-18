import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def _check_dynamic(dynamic, start, end, nobs):
    """
    Verify dynamic and warn or error if issues

    Parameters
    ----------
    dynamic : {int, None}
        The offset relative to start of the dynamic forecasts. None if no
        dynamic forecasts are required.
    start : int
        The location of the first forecast.
    end : int
        The location of the final forecast (inclusive).
    nobs : int
        The number of observations in the time series.

    Returns
    -------
    dynamic : {int, None}
        The start location of the first dynamic forecast. None if there
        are no in-sample dynamic forecasts.
    ndynamic : int
        The number of dynamic forecasts
    """
    if dynamic is None:
        return (dynamic, 0)
    dynamic = start + dynamic
    if dynamic < 0:
        raise ValueError('Dynamic prediction cannot begin prior to the first observation in the sample.')
    elif dynamic > end:
        warn('Dynamic prediction specified to begin after the end of prediction, and so has no effect.', ValueWarning)
        return (None, 0)
    elif dynamic > nobs:
        warn('Dynamic prediction specified to begin during out-of-sample forecasting period, and so has no effect.', ValueWarning)
        return (None, 0)
    ndynamic = max(0, min(end, nobs) - dynamic)
    return (dynamic, ndynamic)