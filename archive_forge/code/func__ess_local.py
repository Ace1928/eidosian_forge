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
def _ess_local(ary, prob, relative=False):
    """Compute the effective sample size for the specific residual."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    if prob is None:
        raise TypeError('Prob not defined.')
    if len(prob) != 2:
        raise ValueError('Prob argument in ess local must be upper and lower bound')
    quantile = _quantile(ary, prob)
    iquantile = (quantile[0] <= ary) & (ary <= quantile[1])
    return _ess(_split_chains(iquantile), relative=relative)