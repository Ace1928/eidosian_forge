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
def _ic_matrix(ics, ic_i):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics[ic_i].iloc[0])
    ic_i_val = np.zeros((rows, cols))
    for idx, val in enumerate(ics.index):
        ic = ics.loc[val][ic_i]
        if len(ic) != rows:
            raise ValueError('The number of observations should be the same across all models')
        ic_i_val[:, idx] = ic
    return (rows, cols, ic_i_val)