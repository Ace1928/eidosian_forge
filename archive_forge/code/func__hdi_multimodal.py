import itertools
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Mapping, cast, Callable
import numpy as np
import pandas as pd
import scipy.stats as st
from xarray_einstats import stats
import xarray as xr
from scipy.optimize import minimize
from typing_extensions import Literal
from .. import _log
from ..data import InferenceData, convert_to_dataset, convert_to_inference_data, extract
from ..rcparams import rcParams, ScaleKeyword, ICKeyword
from ..utils import Numba, _numba_var, _var_names, get_coords
from .density_utils import get_bins as _get_bins
from .density_utils import histogram as _histogram
from .density_utils import kde as _kde
from .diagnostics import _mc_error, _multichain_statistics, ess
from .stats_utils import ELPDData, _circular_standard_deviation, smooth_data
from .stats_utils import get_log_likelihood as _get_log_likelihood
from .stats_utils import get_log_prior as _get_log_prior
from .stats_utils import logsumexp as _logsumexp
from .stats_utils import make_ufunc as _make_ufunc
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
from ..sel_utils import xarray_var_iter
from ..labels import BaseLabeller
def _hdi_multimodal(ary, hdi_prob, skipna, max_modes):
    """Compute HDI if the distribution is multimodal."""
    ary = ary.flatten()
    if skipna:
        ary = ary[~np.isnan(ary)]
    if ary.dtype.kind == 'f':
        bins, density = _kde(ary)
        lower, upper = (bins[0], bins[-1])
        range_x = upper - lower
        dx = range_x / len(density)
    else:
        bins = _get_bins(ary)
        _, density, _ = _histogram(ary, bins=bins)
        dx = np.diff(bins)[0]
    density *= dx
    idx = np.argsort(-density)
    intervals = bins[idx][density[idx].cumsum() <= hdi_prob]
    intervals.sort()
    intervals_splitted = np.split(intervals, np.where(np.diff(intervals) >= dx * 1.1)[0] + 1)
    hdi_intervals = np.full((max_modes, 2), np.nan)
    for i, interval in enumerate(intervals_splitted):
        if i == max_modes:
            warnings.warn(f'found more modes than {max_modes}, returning only the first {max_modes} modes')
            break
        if interval.size == 0:
            hdi_intervals[i] = np.asarray([bins[0], bins[0]])
        else:
            hdi_intervals[i] = np.asarray([interval[0], interval[-1]])
    return np.array(hdi_intervals)