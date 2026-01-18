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
def _preprocess_X(self, X, *, reset):
    """Preprocess and validate X.

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        reset : bool
            Whether to reset the `n_features_in_` and `feature_names_in_ attributes.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        known_categories : list of ndarray of shape (n_categories,)
            List of known categories for each categorical feature.
        """
    check_X_kwargs = dict(dtype=[X_DTYPE], force_all_finite=False)
    if not reset:
        if self._preprocessor is None:
            return self._validate_data(X, reset=False, **check_X_kwargs)
        return self._preprocessor.transform(X)
    self.is_categorical_ = self._check_categorical_features(X)
    if self.is_categorical_ is None:
        self._preprocessor = None
        self._is_categorical_remapped = None
        X = self._validate_data(X, **check_X_kwargs)
        return (X, None)
    n_features = X.shape[1]
    ordinal_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan, dtype=X_DTYPE)
    check_X = partial(check_array, **check_X_kwargs)
    numerical_preprocessor = FunctionTransformer(check_X)
    self._preprocessor = ColumnTransformer([('encoder', ordinal_encoder, self.is_categorical_), ('numerical', numerical_preprocessor, ~self.is_categorical_)])
    self._preprocessor.set_output(transform='default')
    X = self._preprocessor.fit_transform(X)
    known_categories = self._check_categories()
    self.n_features_in_ = self._preprocessor.n_features_in_
    with suppress(AttributeError):
        self.feature_names_in_ = self._preprocessor.feature_names_in_
    categorical_remapped = np.zeros(n_features, dtype=bool)
    categorical_remapped[self._preprocessor.output_indices_['encoder']] = True
    self._is_categorical_remapped = categorical_remapped
    return (X, known_categories)