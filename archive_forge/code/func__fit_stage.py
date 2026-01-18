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
def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc=None, X_csr=None):
    """Fit another stage of ``n_trees_per_iteration_`` trees."""
    original_y = y
    if isinstance(self._loss, HuberLoss):
        set_huber_delta(loss=self._loss, y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight)
    neg_gradient = -self._loss.gradient(y_true=y, raw_prediction=raw_predictions, sample_weight=None)
    if neg_gradient.ndim == 1:
        neg_g_view = neg_gradient.reshape((-1, 1))
    else:
        neg_g_view = neg_gradient
    for k in range(self.n_trees_per_iteration_):
        if self._loss.is_multiclass:
            y = np.array(original_y == k, dtype=np.float64)
        tree = DecisionTreeRegressor(criterion=self.criterion, splitter='best', max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, min_impurity_decrease=self.min_impurity_decrease, max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes, random_state=random_state, ccp_alpha=self.ccp_alpha)
        if self.subsample < 1.0:
            sample_weight = sample_weight * sample_mask.astype(np.float64)
        X = X_csc if X_csc is not None else X
        tree.fit(X, neg_g_view[:, k], sample_weight=sample_weight, check_input=False)
        X_for_tree_update = X_csr if X_csr is not None else X
        _update_terminal_regions(self._loss, tree.tree_, X_for_tree_update, y, neg_g_view[:, k], raw_predictions, sample_weight, sample_mask, learning_rate=self.learning_rate, k=k)
        self.estimators_[i, k] = tree
    return raw_predictions