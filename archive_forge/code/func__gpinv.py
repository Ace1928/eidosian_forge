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
def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        x[probs == 1] = np.inf if kappa >= 0 else -sigma / kappa
    return x