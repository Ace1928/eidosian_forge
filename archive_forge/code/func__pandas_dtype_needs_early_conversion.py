import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    from pandas import SparseDtype
    from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype
    if is_bool_dtype(pd_dtype):
        return True
    if isinstance(pd_dtype, SparseDtype):
        return False
    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False
    if isinstance(pd_dtype, SparseDtype) or not is_extension_array_dtype(pd_dtype):
        return False
    elif is_float_dtype(pd_dtype):
        return True
    elif is_integer_dtype(pd_dtype):
        return True
    return False