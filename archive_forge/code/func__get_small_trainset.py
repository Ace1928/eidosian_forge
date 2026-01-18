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
def _get_small_trainset(self, X_binned_train, y_train, sample_weight_train, seed):
    """Compute the indices of the subsample set and return this set.

        For efficiency, we need to subsample the training set to compute scores
        with scorers.
        """
    subsample_size = 10000
    if X_binned_train.shape[0] > subsample_size:
        indices = np.arange(X_binned_train.shape[0])
        stratify = y_train if is_classifier(self) else None
        indices = resample(indices, n_samples=subsample_size, replace=False, random_state=seed, stratify=stratify)
        X_binned_small_train = X_binned_train[indices]
        y_small_train = y_train[indices]
        if sample_weight_train is not None:
            sample_weight_small_train = sample_weight_train[indices]
        else:
            sample_weight_small_train = None
        X_binned_small_train = np.ascontiguousarray(X_binned_small_train)
        return (X_binned_small_train, y_small_train, sample_weight_small_train, indices)
    else:
        return (X_binned_train, y_train, sample_weight_train, slice(None))