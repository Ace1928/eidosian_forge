import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def _ess_tail(ary, prob=None, relative=False):
    """Compute the effective sample size for the tail.

    If `prob` defined, ess = min(qess(prob), qess(1-prob))
    """
    if prob is None:
        prob = (0.05, 0.95)
    elif not isinstance(prob, Sequence):
        prob = (prob, 1 - prob)
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    prob_low, prob_high = prob
    quantile_low_ess = _ess_quantile(ary, prob_low, relative=relative)
    quantile_high_ess = _ess_quantile(ary, prob_high, relative=relative)
    return min(quantile_low_ess, quantile_high_ess)