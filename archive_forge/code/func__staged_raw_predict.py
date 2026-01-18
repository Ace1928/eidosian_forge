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
def _staged_raw_predict(self, X):
    """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        raw_predictions : generator of ndarray of shape             (n_samples, n_trees_per_iteration)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
    check_is_fitted(self)
    X = self._preprocess_X(X, reset=False)
    if X.shape[1] != self._n_features:
        raise ValueError('X has {} features but this estimator was trained with {} features.'.format(X.shape[1], self._n_features))
    n_samples = X.shape[0]
    raw_predictions = np.zeros(shape=(n_samples, self.n_trees_per_iteration_), dtype=self._baseline_prediction.dtype, order='F')
    raw_predictions += self._baseline_prediction
    n_threads = _openmp_effective_n_threads()
    for iteration in range(len(self._predictors)):
        self._predict_iterations(X, self._predictors[iteration:iteration + 1], raw_predictions, is_binned=False, n_threads=n_threads)
        yield raw_predictions.copy()