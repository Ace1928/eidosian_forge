import itertools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
from numbers import Integral, Real
from time import time
import numpy as np
from ..._loss.loss import (
from ...base import (
from ...compose import ColumnTransformer
from ...metrics import check_scoring
from ...metrics._scorer import _SCORERS
from ...model_selection import train_test_split
from ...preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from ...utils import check_random_state, compute_sample_weight, is_scalar_nan, resample
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ...utils.multiclass import check_classification_targets
from ...utils.validation import (
from ._gradient_boosting import _update_raw_predictions
from .binning import _BinMapper
from .common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from .grower import TreeGrower
def _check_categorical_features(self, X):
    """Check and validate categorical features in X

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        Return
        ------
        is_categorical : ndarray of shape (n_features,) or None, dtype=bool
            Indicates whether a feature is categorical. If no feature is
            categorical, this is None.
        """
    if _is_pandas_df(X):
        X_is_dataframe = True
        categorical_columns_mask = np.asarray(X.dtypes == 'category')
        X_has_categorical_columns = categorical_columns_mask.any()
    elif hasattr(X, '__dataframe__'):
        X_is_dataframe = True
        categorical_columns_mask = np.asarray([c.dtype[0].name == 'CATEGORICAL' for c in X.__dataframe__().get_columns()])
        X_has_categorical_columns = categorical_columns_mask.any()
    else:
        X_is_dataframe = False
        categorical_columns_mask = None
        X_has_categorical_columns = False
    if isinstance(self.categorical_features, str) and self.categorical_features == 'warn':
        if X_has_categorical_columns:
            warnings.warn("The categorical_features parameter will change to 'from_dtype' in v1.6. The 'from_dtype' option automatically treats categorical dtypes in a DataFrame as categorical features.", FutureWarning)
        categorical_features = None
    else:
        categorical_features = self.categorical_features
    categorical_by_dtype = isinstance(categorical_features, str) and categorical_features == 'from_dtype'
    no_categorical_dtype = categorical_features is None or (categorical_by_dtype and (not X_is_dataframe))
    if no_categorical_dtype:
        return None
    use_pandas_categorical = categorical_by_dtype and X_is_dataframe
    if use_pandas_categorical:
        categorical_features = categorical_columns_mask
    else:
        categorical_features = np.asarray(categorical_features)
    if categorical_features.size == 0:
        return None
    if categorical_features.dtype.kind not in ('i', 'b', 'U', 'O'):
        raise ValueError(f'categorical_features must be an array-like of bool, int or str, got: {categorical_features.dtype.name}.')
    if categorical_features.dtype.kind == 'O':
        types = set((type(f) for f in categorical_features))
        if types != {str}:
            raise ValueError(f'categorical_features must be an array-like of bool, int or str, got: {', '.join(sorted((t.__name__ for t in types)))}.')
    n_features = X.shape[1]
    feature_names_in_ = getattr(X, 'columns', None)
    if categorical_features.dtype.kind in ('U', 'O'):
        if feature_names_in_ is None:
            raise ValueError('categorical_features should be passed as an array of integers or as a boolean mask when the model is fitted on data without feature names.')
        is_categorical = np.zeros(n_features, dtype=bool)
        feature_names = list(feature_names_in_)
        for feature_name in categorical_features:
            try:
                is_categorical[feature_names.index(feature_name)] = True
            except ValueError as e:
                raise ValueError(f"categorical_features has a item value '{feature_name}' which is not a valid feature name of the training data. Observed feature names: {feature_names}") from e
    elif categorical_features.dtype.kind == 'i':
        if np.max(categorical_features) >= n_features or np.min(categorical_features) < 0:
            raise ValueError('categorical_features set as integer indices must be in [0, n_features - 1]')
        is_categorical = np.zeros(n_features, dtype=bool)
        is_categorical[categorical_features] = True
    else:
        if categorical_features.shape[0] != n_features:
            raise ValueError(f'categorical_features set as a boolean mask must have shape (n_features,), got: {categorical_features.shape}')
        is_categorical = categorical_features
    if not np.any(is_categorical):
        return None
    return is_categorical