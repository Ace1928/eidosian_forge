import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def _circfunc(samples, high, low, skipna):
    samples = np.asarray(samples)
    if skipna:
        samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan
    return _angle(samples, low, high, np.pi)