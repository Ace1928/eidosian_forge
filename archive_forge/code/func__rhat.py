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
def _rhat(ary):
    """Compute the rhat for a 2d array."""
    _numba_flag = Numba.numba_flag
    ary = np.asarray(ary, dtype=float)
    if _not_valid(ary, check_shape=False):
        return np.nan
    _, num_samples = ary.shape
    chain_mean = np.mean(ary, axis=1)
    chain_var = _numba_var(svar, np.var, ary, axis=1, ddof=1)
    between_chain_variance = num_samples * _numba_var(svar, np.var, chain_mean, axis=None, ddof=1)
    within_chain_variance = np.mean(chain_var)
    rhat_value = np.sqrt((between_chain_variance / within_chain_variance + num_samples - 1) / num_samples)
    return rhat_value