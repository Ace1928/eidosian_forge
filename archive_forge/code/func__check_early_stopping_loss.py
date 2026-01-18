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
def _check_early_stopping_loss(self, raw_predictions, y_train, sample_weight_train, raw_predictions_val, y_val, sample_weight_val, n_threads=1):
    """Check if fitting should be early-stopped based on loss.

        Scores are computed on validation data or on training data.
        """
    self.train_score_.append(-self._loss(y_true=y_train, raw_prediction=raw_predictions, sample_weight=sample_weight_train, n_threads=n_threads))
    if self._use_validation_data:
        self.validation_score_.append(-self._loss(y_true=y_val, raw_prediction=raw_predictions_val, sample_weight=sample_weight_val, n_threads=n_threads))
        return self._should_stop(self.validation_score_)
    else:
        return self._should_stop(self.train_score_)