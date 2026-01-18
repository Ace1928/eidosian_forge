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
def _powerscale_sens(draws, *, lower_weights=None, upper_weights=None, delta=0.01):
    """
    Calculate power-scaling sensitivity by finite difference
    second derivative of CJS
    """
    lower_cjs = max(_cjs_dist(draws=draws, weights=lower_weights), _cjs_dist(draws=-1 * draws, weights=lower_weights))
    upper_cjs = max(_cjs_dist(draws=draws, weights=upper_weights), _cjs_dist(draws=-1 * draws, weights=upper_weights))
    logdiffsquare = 2 * np.log2(1 + delta)
    grad = (lower_cjs + upper_cjs) / logdiffsquare
    return grad