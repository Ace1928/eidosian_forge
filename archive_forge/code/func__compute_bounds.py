from __future__ import annotations
from packaging.version import Version
import inspect
import warnings
import os
from math import isnan
import numpy as np
import pandas as pd
import xarray as xr
from datashader.utils import Expr, ngjit
from datashader.macros import expand_varargs
@staticmethod
def _compute_bounds(s):
    if cudf and isinstance(s, cudf.Series):
        s = s.nans_to_nulls()
        return (s.min(), s.max())
    elif isinstance(s, pd.Series):
        return Glyph._compute_bounds_numba(s.values)
    elif isinstance(s, xr.DataArray):
        if cp and isinstance(s.data, cp.ndarray):
            return (s.min().item(), s.max().item())
        else:
            return Glyph._compute_bounds_numba(s.values.ravel())
    else:
        return Glyph._compute_bounds_numba(s)