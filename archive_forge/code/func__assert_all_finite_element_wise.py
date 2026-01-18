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
def _assert_all_finite_element_wise(X, *, xp, allow_nan, msg_dtype=None, estimator_name=None, input_name=''):
    use_cython = xp is np and X.data.contiguous and (X.dtype.type in {np.float32, np.float64})
    if use_cython:
        out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
        has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
        has_inf = out == FiniteStatus.has_infinite
    else:
        has_inf = xp.any(xp.isinf(X))
        has_nan_error = False if allow_nan else xp.any(xp.isnan(X))
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = 'NaN'
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f'infinity or a value too large for {msg_dtype!r}'
        padded_input_name = input_name + ' ' if input_name else ''
        msg_err = f'Input {padded_input_name}contains {type_err}.'
        if estimator_name and input_name == 'X' and has_nan_error:
            msg_err += f'\n{estimator_name} does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values'
        raise ValueError(msg_err)