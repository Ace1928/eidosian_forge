import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
def _boost(self, iboost, X, y, sample_weight, random_state):
    """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
            Controls also the bootstrap of the weights used to train the weak
            learner.
            replacement.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
    estimator = self._make_estimator(random_state=random_state)
    bootstrap_idx = random_state.choice(np.arange(_num_samples(X)), size=_num_samples(X), replace=True, p=sample_weight)
    X_ = _safe_indexing(X, bootstrap_idx)
    y_ = _safe_indexing(y, bootstrap_idx)
    estimator.fit(X_, y_)
    y_predict = estimator.predict(X)
    error_vect = np.abs(y_predict - y)
    sample_mask = sample_weight > 0
    masked_sample_weight = sample_weight[sample_mask]
    masked_error_vector = error_vect[sample_mask]
    error_max = masked_error_vector.max()
    if error_max != 0:
        masked_error_vector /= error_max
    if self.loss == 'square':
        masked_error_vector **= 2
    elif self.loss == 'exponential':
        masked_error_vector = 1.0 - np.exp(-masked_error_vector)
    estimator_error = (masked_sample_weight * masked_error_vector).sum()
    if estimator_error <= 0:
        return (sample_weight, 1.0, 0.0)
    elif estimator_error >= 0.5:
        if len(self.estimators_) > 1:
            self.estimators_.pop(-1)
        return (None, None, None)
    beta = estimator_error / (1.0 - estimator_error)
    estimator_weight = self.learning_rate * np.log(1.0 / beta)
    if not iboost == self.n_estimators - 1:
        sample_weight[sample_mask] *= np.power(beta, (1.0 - masked_error_vector) * self.learning_rate)
    return (sample_weight, estimator_weight, estimator_error)