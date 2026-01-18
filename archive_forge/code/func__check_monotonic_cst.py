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
def _check_monotonic_cst(estimator, monotonic_cst=None):
    """Check the monotonic constraints and return the corresponding array.

    This helper function should be used in the `fit` method of an estimator
    that supports monotonic constraints and called after the estimator has
    introspected input data to set the `n_features_in_` and optionally the
    `feature_names_in_` attributes.

    .. versionadded:: 1.2

    Parameters
    ----------
    estimator : estimator instance

    monotonic_cst : array-like of int, dict of str or None, default=None
        Monotonic constraints for the features.

        - If array-like, then it should contain only -1, 0 or 1. Each value
            will be checked to be in [-1, 0, 1]. If a value is -1, then the
            corresponding feature is required to be monotonically decreasing.
        - If dict, then it the keys should be the feature names occurring in
            `estimator.feature_names_in_` and the values should be -1, 0 or 1.
        - If None, then an array of 0s will be allocated.

    Returns
    -------
    monotonic_cst : ndarray of int
        Monotonic constraints for each feature.
    """
    original_monotonic_cst = monotonic_cst
    if monotonic_cst is None or isinstance(monotonic_cst, dict):
        monotonic_cst = np.full(shape=estimator.n_features_in_, fill_value=0, dtype=np.int8)
        if isinstance(original_monotonic_cst, dict):
            if not hasattr(estimator, 'feature_names_in_'):
                raise ValueError(f'{estimator.__class__.__name__} was not fitted on data with feature names. Pass monotonic_cst as an integer array instead.')
            unexpected_feature_names = list(set(original_monotonic_cst) - set(estimator.feature_names_in_))
            unexpected_feature_names.sort()
            n_unexpeced = len(unexpected_feature_names)
            if unexpected_feature_names:
                if len(unexpected_feature_names) > 5:
                    unexpected_feature_names = unexpected_feature_names[:5]
                    unexpected_feature_names.append('...')
                raise ValueError(f'monotonic_cst contains {n_unexpeced} unexpected feature names: {unexpected_feature_names}.')
            for feature_idx, feature_name in enumerate(estimator.feature_names_in_):
                if feature_name in original_monotonic_cst:
                    cst = original_monotonic_cst[feature_name]
                    if cst not in [-1, 0, 1]:
                        raise ValueError(f"monotonic_cst['{feature_name}'] must be either -1, 0 or 1. Got {cst!r}.")
                    monotonic_cst[feature_idx] = cst
    else:
        unexpected_cst = np.setdiff1d(monotonic_cst, [-1, 0, 1])
        if unexpected_cst.shape[0]:
            raise ValueError(f'monotonic_cst must be an array-like of -1, 0 or 1. Observed values: {unexpected_cst.tolist()}.')
        monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)
        if monotonic_cst.shape[0] != estimator.n_features_in_:
            raise ValueError(f'monotonic_cst has shape {monotonic_cst.shape} but the input data X has {estimator.n_features_in_} features.')
    return monotonic_cst