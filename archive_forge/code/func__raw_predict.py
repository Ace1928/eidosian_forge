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
def _raw_predict(self, X, n_threads=None):
    """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        n_threads : int, default=None
            Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
            to determine the effective number of threads use, which takes cgroups CPU
            quotes into account. See the docstring of `_openmp_effective_n_threads`
            for details.

        Returns
        -------
        raw_predictions : array, shape (n_samples, n_trees_per_iteration)
            The raw predicted values.
        """
    check_is_fitted(self)
    is_binned = getattr(self, '_in_fit', False)
    if not is_binned:
        X = self._preprocess_X(X, reset=False)
    n_samples = X.shape[0]
    raw_predictions = np.zeros(shape=(n_samples, self.n_trees_per_iteration_), dtype=self._baseline_prediction.dtype, order='F')
    raw_predictions += self._baseline_prediction
    n_threads = _openmp_effective_n_threads(n_threads)
    self._predict_iterations(X, self._predictors, raw_predictions, is_binned, n_threads)
    return raw_predictions