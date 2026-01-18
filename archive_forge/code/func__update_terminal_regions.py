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
def _update_terminal_regions(loss, tree, X, y, neg_gradient, raw_prediction, sample_weight, sample_mask, learning_rate=0.1, k=0):
    """Update the leaf values to be predicted by the tree and raw_prediction.

    The current raw predictions of the model (of this stage) are updated.

    Additionally, the terminal regions (=leaves) of the given tree are updated as well.
    This corresponds to the line search step in "Greedy Function Approximation" by
    Friedman, Algorithm 1 step 5.

    Update equals:
        argmin_{x} loss(y_true, raw_prediction_old + x * tree.value)

    For non-trivial cases like the Binomial loss, the update has no closed formula and
    is an approximation, again, see the Friedman paper.

    Also note that the update formula for the SquaredError is the identity. Therefore,
    in this case, the leaf values don't need an update and only the raw_predictions are
    updated (with the learning rate included).

    Parameters
    ----------
    loss : BaseLoss
    tree : tree.Tree
        The tree object.
    X : ndarray of shape (n_samples, n_features)
        The data array.
    y : ndarray of shape (n_samples,)
        The target labels.
    neg_gradient : ndarray of shape (n_samples,)
        The negative gradient.
    raw_prediction : ndarray of shape (n_samples, n_trees_per_iteration)
        The raw predictions (i.e. values from the tree leaves) of the
        tree ensemble at iteration ``i - 1``.
    sample_weight : ndarray of shape (n_samples,)
        The weight of each sample.
    sample_mask : ndarray of shape (n_samples,)
        The sample mask to be used.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by
         ``learning_rate``.
    k : int, default=0
        The index of the estimator being updated.
    """
    terminal_regions = tree.apply(X)
    if not isinstance(loss, HalfSquaredError):
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1
        if isinstance(loss, HalfBinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                numerator = np.average(neg_g, weights=sw)
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)
        elif isinstance(loss, HalfMultinomialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                neg_g = neg_gradient.take(indices, axis=0)
                prob = y_ - neg_g
                K = loss.n_classes
                numerator = np.average(neg_g, weights=sw)
                numerator *= (K - 1) / K
                denominator = np.average(prob * (1 - prob), weights=sw)
                return _safe_divide(numerator, denominator)
        elif isinstance(loss, ExponentialLoss):

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                neg_g = neg_gradient.take(indices, axis=0)
                numerator = np.average(neg_g, weights=sw)
                hessian = neg_g.copy()
                hessian[y_ == 0] *= -1
                denominator = np.average(hessian, weights=sw)
                return _safe_divide(numerator, denominator)
        else:

            def compute_update(y_, indices, neg_gradient, raw_prediction, k):
                return loss.fit_intercept_only(y_true=y_ - raw_prediction[indices, k], sample_weight=sw)
        for leaf in np.nonzero(tree.children_left == TREE_LEAF)[0]:
            indices = np.nonzero(masked_terminal_regions == leaf)[0]
            y_ = y.take(indices, axis=0)
            sw = None if sample_weight is None else sample_weight[indices]
            update = compute_update(y_, indices, neg_gradient, raw_prediction, k)
            tree.value[leaf, 0, 0] = update
    raw_prediction[:, k] += learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)