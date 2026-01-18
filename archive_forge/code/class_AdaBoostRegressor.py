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
class AdaBoostRegressor(_RoutingNotSupportedMixin, RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration. A higher
        learning rate increases the contribution of each regressor. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of regressors
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostClassifier : An AdaBoost classifier.
    GradientBoostingRegressor : Gradient Boosting Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...
    """
    _parameter_constraints: dict = {**BaseWeightBoosting._parameter_constraints, 'loss': [StrOptions({'linear', 'square', 'exponential'})]}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        self.loss = loss
        self.random_state = random_state

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor(max_depth=3))

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

    def _get_median_predict(self, X, limit):
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T
        sorted_idx = np.argsort(predictions, axis=1)
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)