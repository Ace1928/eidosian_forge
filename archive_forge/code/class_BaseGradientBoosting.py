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
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""
    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints, 'learning_rate': [Interval(Real, 0.0, None, closed='left')], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'criterion': [StrOptions({'friedman_mse', 'squared_error'})], 'subsample': [Interval(Real, 0.0, 1.0, closed='right')], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'validation_fraction': [Interval(Real, 0.0, 1.0, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0.0, None, closed='left')]}
    _parameter_constraints.pop('splitter')
    _parameter_constraints.pop('monotonic_cst')

    @abstractmethod
    def __init__(self, *, loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, init, subsample, max_features, ccp_alpha, random_state, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    @abstractmethod
    def _encode_y(self, y=None, sample_weight=None):
        """Called by fit to validate and encode y."""

    @abstractmethod
    def _get_loss(self, sample_weight):
        """Get loss object from sklearn._loss.loss."""

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

    def _init_state(self):
        """Initialize model state and allocate model state data structures."""
        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy='prior')
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy='quantile', quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy='quantile', quantile=self.alpha)
            else:
                self.init_ = DummyRegressor(strategy='mean')
        self.estimators_ = np.empty((self.n_estimators, self.n_trees_per_iteration_), dtype=object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros(self.n_estimators, dtype=np.float64)
            self.oob_scores_ = np.zeros(self.n_estimators, dtype=np.float64)
            self.oob_score_ = np.nan

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'oob_scores_'):
            del self.oob_scores_
        if hasattr(self, 'oob_score_'):
            del self.oob_score_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng

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

    def _is_fitted(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshotting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not self.warm_start:
            self._clear_state()
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE, multi_output=True)
        sample_weight_is_none = sample_weight is None
        sample_weight = _check_sample_weight(sample_weight, X)
        if sample_weight_is_none:
            y = self._encode_y(y=y, sample_weight=None)
        else:
            y = self._encode_y(y=y, sample_weight=sample_weight)
        y = column_or_1d(y, warn=True)
        self._set_max_features()
        self._loss = self._get_loss(sample_weight=sample_weight)
        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X_train, X_val, y_train, y_val, sample_weight_train, sample_weight_val = train_test_split(X, y, sample_weight, random_state=self.random_state, test_size=self.validation_fraction, stratify=stratify)
            if is_classifier(self):
                if self.n_classes_ != np.unique(y_train).shape[0]:
                    raise ValueError('The training data after the early stopping split is missing some classes. Try using another random seed.')
        else:
            X_train, y_train, sample_weight_train = (X, y, sample_weight)
            X_val = y_val = sample_weight_val = None
        n_samples = X_train.shape[0]
        if not self._is_fitted():
            self._init_state()
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(n_samples, self.n_trees_per_iteration_), dtype=np.float64)
            else:
                if sample_weight_is_none:
                    self.init_.fit(X_train, y_train)
                else:
                    msg = 'The initial estimator {} does not support sample weights.'.format(self.init_.__class__.__name__)
                    try:
                        self.init_.fit(X_train, y_train, sample_weight=sample_weight_train)
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            raise ValueError(msg) from e
                        else:
                            raise
                    except ValueError as e:
                        if 'pass parameters to specific steps of your pipeline using the stepname__parameter' in str(e):
                            raise ValueError(msg) from e
                        else:
                            raise
                raw_predictions = _init_raw_predictions(X_train, self.init_, self._loss, is_classifier(self))
            begin_at_stage = 0
            self._rng = check_random_state(self.random_state)
        else:
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to estimators_.shape[0]=%d when warm_start==True' % (self.n_estimators, self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            X_train = check_array(X_train, dtype=DTYPE, order='C', accept_sparse='csr', force_all_finite=False)
            raw_predictions = self._raw_predict(X_train)
            self._resize_state()
        n_stages = self._fit_stages(X_train, y_train, raw_predictions, sample_weight_train, self._rng, X_val, y_val, sample_weight_val, begin_at_stage, monitor)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]
        self.n_estimators_ = n_stages
        return self

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage=0, monitor=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)
        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None
        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)
        if isinstance(self._loss, (HalfSquaredError, HalfBinomialLoss)):
            factor = 2
        else:
            factor = 1
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                y_oob_masked = y[~sample_mask]
                sample_weight_oob_masked = sample_weight[~sample_mask]
                if i == 0:
                    initial_loss = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
            raw_predictions = self._fit_stage(i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc=X_csc, X_csr=X_csr)
            if do_oob:
                self.train_score_[i] = factor * self._loss(y_true=y[sample_mask], raw_prediction=raw_predictions[sample_mask], sample_weight=sample_weight[sample_mask])
                self.oob_scores_[i] = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                self.train_score_[i] = factor * self._loss(y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight)
            if self.verbose > 0:
                verbose_reporter.update(i, self)
            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break
            if self.n_iter_no_change is not None:
                validation_loss = factor * self._loss(y_val, next(y_val_pred_iter), sample_weight_val)
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break
        return i + 1

    def _make_estimator(self, append=True):
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64)
        else:
            raw_predictions = _init_raw_predictions(X, self.init_, self._loss, is_classifier(self))
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            If False, the input arrays X will not be checked.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, order='C', accept_sparse='csr', reset=False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
            yield raw_predictions.copy()

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
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        self._check_initialized()
        relevant_trees = [tree for stage in self.estimators_ for tree in stage if tree.tree_.node_count > 1]
        if not relevant_trees:
            return np.zeros(shape=self.n_features_in_, dtype=np.float64)
        relevant_feature_importances = [tree.tree_.compute_feature_importances(normalize=False) for tree in relevant_trees]
        avg_feature_importances = np.mean(relevant_feature_importances, axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features,)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape                 (n_trees_per_iteration_, n_samples)
            The value of the partial dependence function on each grid point.
        """
        if self.init is not None:
            warnings.warn('Using recursion method with a non-constant init predictor will lead to incorrect partial dependence values. Got init=%s.' % self.init, UserWarning)
        grid = np.asarray(grid, dtype=DTYPE, order='C')
        n_estimators, n_trees_per_stage = self.estimators_.shape
        averaged_predictions = np.zeros((n_trees_per_stage, grid.shape[0]), dtype=np.float64, order='C')
        for stage in range(n_estimators):
            for k in range(n_trees_per_stage):
                tree = self.estimators_[stage, k].tree_
                tree.compute_partial_dependence(grid, target_features, averaged_predictions[k])
        averaged_predictions *= self.learning_rate
        return averaged_predictions

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))
        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)
        return leaves