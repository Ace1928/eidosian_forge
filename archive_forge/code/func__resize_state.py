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
def _resize_state(self):
    """Add additional ``n_estimators`` entries to all attributes."""
    total_n_estimators = self.n_estimators
    if total_n_estimators < self.estimators_.shape[0]:
        raise ValueError('resize with smaller n_estimators %d < %d' % (total_n_estimators, self.estimators_[0]))
    self.estimators_ = np.resize(self.estimators_, (total_n_estimators, self.n_trees_per_iteration_))
    self.train_score_ = np.resize(self.train_score_, total_n_estimators)
    if self.subsample < 1 or hasattr(self, 'oob_improvement_'):
        if hasattr(self, 'oob_improvement_'):
            self.oob_improvement_ = np.resize(self.oob_improvement_, total_n_estimators)
            self.oob_scores_ = np.resize(self.oob_scores_, total_n_estimators)
            self.oob_score_ = np.nan
        else:
            self.oob_improvement_ = np.zeros((total_n_estimators,), dtype=np.float64)
            self.oob_scores_ = np.zeros((total_n_estimators,), dtype=np.float64)
            self.oob_score_ = np.nan