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
def _check_feature_names_in(estimator, input_features=None, *, generate_names=True):
    """Check `input_features` and generate names if needed.

    Commonly used in :term:`get_feature_names_out`.

    Parameters
    ----------
    input_features : array-like of str or None, default=None
        Input features.

        - If `input_features` is `None`, then `feature_names_in_` is
          used as feature names in. If `feature_names_in_` is not defined,
          then the following input feature names are generated:
          `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
        - If `input_features` is an array-like, then `input_features` must
          match `feature_names_in_` if `feature_names_in_` is defined.

    generate_names : bool, default=True
        Whether to generate names when `input_features` is `None` and
        `estimator.feature_names_in_` is not defined. This is useful for transformers
        that validates `input_features` but do not require them in
        :term:`get_feature_names_out` e.g. `PCA`.

    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """
    feature_names_in_ = getattr(estimator, 'feature_names_in_', None)
    n_features_in_ = getattr(estimator, 'n_features_in_', None)
    if input_features is not None:
        input_features = np.asarray(input_features, dtype=object)
        if feature_names_in_ is not None and (not np.array_equal(feature_names_in_, input_features)):
            raise ValueError('input_features is not equal to feature_names_in_')
        if n_features_in_ is not None and len(input_features) != n_features_in_:
            raise ValueError(f'input_features should have length equal to number of features ({n_features_in_}), got {len(input_features)}')
        return input_features
    if feature_names_in_ is not None:
        return feature_names_in_
    if not generate_names:
        return
    if n_features_in_ is None:
        raise ValueError('Unable to generate feature names without n_features_in_')
    return np.asarray([f'x{i}' for i in range(n_features_in_)], dtype=object)