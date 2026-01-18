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
def _call_nearest(self, x_new):
    """ Find nearest neighbor interpolated y_new = f(x_new)."""
    x_new_indices = searchsorted(self.x_bds, x_new, side=self._side)
    x_new_indices = x_new_indices.clip(0, len(self.x) - 1).astype(intp)
    y_new = self._y[x_new_indices]
    return y_new