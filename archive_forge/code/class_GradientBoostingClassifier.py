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
class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.

    This algorithm builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage ``n_classes_`` regression trees are fit on the negative gradient
    of the loss function, e.g. binary or multiclass log loss. Binary
    classification is a special case where only a single regression tree is
    induced.

    :class:`sklearn.ensemble.HistGradientBoostingClassifier` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'log_loss', 'exponential'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to binomial and
        multinomial deviance, the same as used in logistic regression.
        It is a good choice for classification with probabilistic outputs.
        For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        'friedman_mse' for the mean squared error with improvement score by
        Friedman, 'squared_error' for mean squared error. The default value of
        'friedman_mse' is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If `None`, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range `[1, inf)`.
        See
        :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    n_trees_per_iteration_ : int
        The number of trees that are built at each iteration. For binary classifiers,
        this is always 1.

        .. versionadded:: 1.4.0

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

        .. versionadded:: 1.3

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    init_ : estimator
        The estimator that provides the initial predictions. Set via the ``init``
        argument.

    estimators_ : ndarray of DecisionTreeRegressor of             shape (n_estimators, ``n_trees_per_iteration_``)
        The collection of fitted sub-estimators. ``n_trees_per_iteration_`` is 1 for
        binary classification, otherwise ``n_classes``.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_classes_ : int
        The number of classes.

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingClassifier : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    The following example shows how to fit a gradient boosting classifier with
    100 decision stumps as weak learners.

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.913...
    """
    _parameter_constraints: dict = {**BaseGradientBoosting._parameter_constraints, 'loss': [StrOptions({'log_loss', 'exponential'})], 'init': [StrOptions({'zero'}), None, HasMethods(['fit', 'predict_proba'])]}

    def __init__(self, *, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, init=init, subsample=subsample, max_features=max_features, random_state=random_state, verbose=verbose, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    def _encode_y(self, y, sample_weight):
        check_classification_targets(y)
        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False)
        self.n_classes_ = n_classes
        if sample_weight is None:
            n_trim_classes = n_classes
        else:
            n_trim_classes = np.count_nonzero(np.bincount(encoded_y_int, sample_weight))
        if n_trim_classes < 2:
            raise ValueError('y contains %d class after sample_weight trimmed classes with zero weights, while a minimum of 2 classes are required.' % n_trim_classes)
        return encoded_y

    def _get_loss(self, sample_weight):
        if self.loss == 'log_loss':
            if self.n_classes_ == 2:
                return HalfBinomialLoss(sample_weight=sample_weight)
            else:
                return HalfMultinomialLoss(sample_weight=sample_weight, n_classes=self.n_classes_)
        elif self.loss == 'exponential':
            if self.n_classes_ > 2:
                raise ValueError(f"loss='{self.loss}' is only suitable for a binary classification problem, you have n_classes={self.n_classes_}. Please use loss='log_loss' instead.")
            else:
                return ExponentialLoss(sample_weight=sample_weight)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        X = self._validate_data(X, dtype=DTYPE, order='C', accept_sparse='csr', reset=False)
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
        return self.classes_[encoded_classes]

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        if self.n_classes_ == 2:
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = (raw_predictions.squeeze() >= 0).astype(int)
                yield self.classes_.take(encoded_classes, axis=0)
        else:
            for raw_predictions in self._staged_raw_predict(X):
                encoded_classes = np.argmax(raw_predictions, axis=1)
                yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        raw_predictions = self.decision_function(X)
        return self._loss.predict_proba(raw_predictions)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss.predict_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError('loss=%r does not support predict_proba' % self.loss) from e