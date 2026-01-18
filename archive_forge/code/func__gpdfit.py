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
def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary: array
        sorted 1D data array

    Returns
    -------
    k: float
        estimated shape parameter
    sigma: float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]
    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    weights /= weights.sum()
    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * ary).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return (k_post, sigma)