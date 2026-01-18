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
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict']), None], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'learning_rate': [Interval(Real, 0, None, closed='neither')], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, estimator=None, *, n_estimators=50, estimator_params=tuple(), learning_rate=1.0, random_state=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params)
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        return self._validate_data(X, accept_sparse=['csr', 'csc'], ensure_2d=True, allow_nd=True, dtype=None, reset=False)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True, allow_nd=True, dtype=None, y_numeric=is_regressor(self))
        sample_weight = _check_sample_weight(sample_weight, X, np.float64, copy=True, only_non_negative=True)
        sample_weight /= sample_weight.sum()
        self._validate_estimator()
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps
        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            sample_weight[zero_weight_mask] = 0.0
            sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight, random_state)
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            if estimator_error == 0:
                break
            sample_weight_sum = np.sum(sample_weight)
            if not np.isfinite(sample_weight_sum):
                warnings.warn(f'Sample weights have reached infinite values, at iteration {iboost}, causing overflow. Iterations stopped. Try lowering the learning rate.', stacklevel=2)
                break
            if sample_weight_sum <= 0:
                break
            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum
        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        X = self._check_X(X)
        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError('Estimator not fitted, call `fit` before `feature_importances_`.')
        try:
            norm = self.estimator_weights_.sum()
            return sum((weight * clf.feature_importances_ for weight, clf in zip(self.estimator_weights_, self.estimators_))) / norm
        except AttributeError as e:
            raise AttributeError('Unable to compute feature importances since estimator does not have a feature_importances_ attribute') from e