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
def _assert_all_finite(X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=''):
    """Like assert_all_finite, but only for ndarray."""
    xp, _ = get_namespace(X)
    if _get_config()['assume_finite']:
        return
    X = xp.asarray(X)
    if X.dtype == np.dtype('object') and (not allow_nan):
        if _object_dtype_isnan(X).any():
            raise ValueError('Input contains NaN')
    if not xp.isdtype(X.dtype, ('real floating', 'complex floating')):
        return
    with np.errstate(over='ignore'):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    _assert_all_finite_element_wise(X, xp=xp, allow_nan=allow_nan, msg_dtype=msg_dtype, estimator_name=estimator_name, input_name=input_name)