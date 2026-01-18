from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def _asdataarray(x_or_q, dim_name):
    """Ensure input is a DataArray.

    This is designed for the x or q arguments in univariate distributions.
    It is also used in multivariate normal distribution but only as a fallback.
    """
    if isinstance(x_or_q, xr.DataArray):
        return x_or_q
    x_or_q_ary = np.asarray(x_or_q)
    if x_or_q_ary.ndim == 0:
        return xr.DataArray(x_or_q_ary)
    if x_or_q_ary.ndim == 1:
        return xr.DataArray(x_or_q_ary, dims=[dim_name], coords={dim_name: np.asarray(x_or_q)})
    raise ValueError('To evaluate distribution methods on data with >=2 dims, the input needs to be a xarray.DataArray')