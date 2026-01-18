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
def _mcse_mean(ary):
    """Compute the Markov Chain mean error."""
    _numba_flag = Numba.numba_flag
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ess = _ess_mean(ary)
    if _numba_flag:
        sd = _sqrt(svar(np.ravel(ary), ddof=1), np.zeros(1))
    else:
        sd = np.std(ary, ddof=1)
    mcse_mean_value = sd / np.sqrt(ess)
    return mcse_mean_value