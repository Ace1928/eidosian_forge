import math
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from .._loss.loss import (
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from ..dummy import DummyClassifier, DummyRegressor
from ..exceptions import NotFittedError
from ..model_selection import train_test_split
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..tree._tree import DOUBLE, DTYPE, TREE_LEAF
from ..utils import check_array, check_random_state, column_or_1d
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.stats import _weighted_percentile
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import BaseEnsemble
from ._gradient_boosting import _random_sample_mask, predict_stage, predict_stages
def _raw_predict_init(self, X):
    """Check input and compute raw predictions of the init estimator."""
    self._check_initialized()
    X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
    if self.init_ == 'zero':
        raw_predictions = np.zeros(shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64)
    else:
        raw_predictions = _init_raw_predictions(X, self.init_, self._loss, is_classifier(self))
    return raw_predictions