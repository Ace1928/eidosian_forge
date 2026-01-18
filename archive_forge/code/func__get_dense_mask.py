from contextlib import suppress
import numpy as np
from scipy import sparse as sp
from . import is_scalar_nan
from .fixes import _object_dtype_isnan
def _get_dense_mask(X, value_to_mask):
    with suppress(ImportError, AttributeError):
        import pandas
        if value_to_mask is pandas.NA:
            return pandas.isna(X)
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == 'f':
            Xt = np.isnan(X)
        elif X.dtype.kind in ('i', 'u'):
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            Xt = _object_dtype_isnan(X)
    else:
        Xt = X == value_to_mask
    return Xt