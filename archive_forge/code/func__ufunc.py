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
def _ufunc(*args, out=None, out_shape=None, **kwargs):
    """General ufunc for single-output function."""
    arys = args[:n_input]
    n_dims_out = None
    if out is None:
        if out_shape is None:
            out = np.empty(arys[-1].shape[:-n_dims])
        else:
            out = np.empty((*arys[-1].shape[:-n_dims], *out_shape))
            n_dims_out = -len(out_shape)
    elif check_shape:
        if out.shape != arys[-1].shape[:-n_dims]:
            msg = f'Shape incorrect for `out`: {out.shape}.'
            msg += f' Correct shape is {arys[-1].shape[:-n_dims]}'
            raise TypeError(msg)
    for idx in np.ndindex(out.shape[:n_dims_out]):
        arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
        out_idx = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
        if n_dims_out is None:
            out_idx = out_idx.item()
        out[idx] = out_idx
    return out