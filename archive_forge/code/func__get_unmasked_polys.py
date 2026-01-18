import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def _get_unmasked_polys(self):
    """Get the unmasked regions using the coordinates and array"""
    mask = np.any(np.ma.getmaskarray(self._coordinates), axis=-1)
    mask = mask[0:-1, 0:-1] | mask[1:, 1:] | mask[0:-1, 1:] | mask[1:, 0:-1]
    if getattr(self, '_deprecated_compression', False) and np.any(self._original_mask):
        return ~(mask | self._original_mask)
    with cbook._setattr_cm(self, _deprecated_compression=False):
        arr = self.get_array()
    if arr is not None:
        arr = np.ma.getmaskarray(arr)
        if arr.ndim == 3:
            mask |= np.any(arr, axis=-1)
        elif arr.ndim == 2:
            mask |= arr
        else:
            mask |= arr.reshape(self._coordinates[:-1, :-1, :].shape[:2])
    return ~mask