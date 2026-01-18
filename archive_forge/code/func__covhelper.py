import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """
    Private function for the computation of covariance and correlation
    coefficients.

    """
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    if not allow_masked and xmask.any():
        raise ValueError('Cannot process masked data.')
    if x.shape[0] == 1:
        rowvar = True
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    if y is None:
        xnotmask = np.logical_not(xmask).astype(int)
    else:
        y = array(y, copy=False, ndmin=2, dtype=float)
        ymask = ma.getmaskarray(y)
        if not allow_masked and ymask.any():
            raise ValueError('Cannot process masked data.')
        if xmask.any() or ymask.any():
            if y.shape == x.shape:
                common_mask = np.logical_or(xmask, ymask)
                if common_mask is not nomask:
                    xmask = x._mask = y._mask = ymask = common_mask
                    x._sharedmask = False
                    y._sharedmask = False
        x = ma.concatenate((x, y), axis)
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(int)
    x -= x.mean(axis=rowvar)[tup]
    return (x, xnotmask, rowvar)