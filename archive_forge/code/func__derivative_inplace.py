from math import prod
import warnings
import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
import scipy.special as spec
from scipy.special import comb
from . import _fitpack_py
from . import dfitpack
from ._polyint import _Interpolator1D
from . import _ppoly
from .interpnd import _ndim_coords_from_arrays
from ._bsplines import make_interp_spline, BSpline
import itertools  # noqa: F401
def _derivative_inplace(self, nu, axis):
    """
        Compute 1-D derivative along a selected dimension in-place
        May result to non-contiguous c array.
        """
    if nu < 0:
        return self._antiderivative_inplace(-nu, axis)
    ndim = len(self.x)
    axis = axis % ndim
    if nu == 0:
        return
    else:
        sl = [slice(None)] * ndim
        sl[axis] = slice(None, -nu, None)
        c2 = self.c[tuple(sl)]
    if c2.shape[axis] == 0:
        shp = list(c2.shape)
        shp[axis] = 1
        c2 = np.zeros(shp, dtype=c2.dtype)
    factor = spec.poch(np.arange(c2.shape[axis], 0, -1), nu)
    sl = [None] * c2.ndim
    sl[axis] = slice(None)
    c2 *= factor[tuple(sl)]
    self.c = c2