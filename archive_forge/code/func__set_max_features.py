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
def _set_max_features(self):
    """Set self.max_features_."""
    if isinstance(self.max_features, str):
        if self.max_features == 'auto':
            if is_classifier(self):
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:
                max_features = self.n_features_in_
        elif self.max_features == 'sqrt':
            max_features = max(1, int(np.sqrt(self.n_features_in_)))
        else:
            max_features = max(1, int(np.log2(self.n_features_in_)))
    elif self.max_features is None:
        max_features = self.n_features_in_
    elif isinstance(self.max_features, Integral):
        max_features = self.max_features
    else:
        max_features = max(1, int(self.max_features * self.n_features_in_))
    self.max_features_ = max_features