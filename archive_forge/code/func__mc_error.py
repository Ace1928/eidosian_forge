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
def _mc_error(ary, batches=5, circular=False):
    """Calculate the simulation standard error, accounting for non-independent samples.

    The trace is divided into batches, and the standard deviation of the batch
    means is calculated.

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples
    batches : integer
        Number of batches
    circular : bool
        Whether to compute the error taking into account `ary` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).

    Returns
    -------
    mc_error : float
        Simulation standard error
    """
    _numba_flag = Numba.numba_flag
    if ary.ndim > 1:
        dims = np.shape(ary)
        trace = np.transpose([t.ravel() for t in ary])
        return np.reshape([_mc_error(t, batches) for t in trace], dims[1:])
    else:
        if _not_valid(ary, check_shape=False):
            return np.nan
        if batches == 1:
            if circular:
                if _numba_flag:
                    std = _circular_standard_deviation(ary, high=np.pi, low=-np.pi)
                else:
                    std = stats.circstd(ary, high=np.pi, low=-np.pi)
            elif _numba_flag:
                std = float(_sqrt(svar(ary), np.zeros(1)).item())
            else:
                std = np.std(ary)
            return std / np.sqrt(len(ary))
        batched_traces = np.resize(ary, (batches, int(len(ary) / batches)))
        if circular:
            means = stats.circmean(batched_traces, high=np.pi, low=-np.pi, axis=1)
            if _numba_flag:
                std = _circular_standard_deviation(means, high=np.pi, low=-np.pi)
            else:
                std = stats.circstd(means, high=np.pi, low=-np.pi)
        else:
            means = np.mean(batched_traces, 1)
            std = _sqrt(svar(means), np.zeros(1)) if _numba_flag else np.std(means)
        return std / np.sqrt(batches)