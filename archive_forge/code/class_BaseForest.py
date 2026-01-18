import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
class BaseForest(MultiOutputMixin, BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'n_estimators': [Interval(Integral, 1, None, closed='left')], 'bootstrap': ['boolean'], 'oob_score': ['boolean', callable], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'max_samples': [None, Interval(RealNotInt, 0.0, 1.0, closed='right'), Interval(Integral, 1, None, closed='left')]}

    @abstractmethod
    def __init__(self, estimator, n_estimators=100, *, estimator_params=tuple(), bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, max_samples=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params)
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')((delayed(tree.apply)(X, check_input=False) for tree in self.estimators_))
        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')((delayed(tree.decision_path)(X, check_input=False) for tree in self.estimators_))
        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()
        return (sparse_hstack(indicators).tocsr(), n_nodes_ptr)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if issparse(y):
            raise ValueError('sparse multilabel-indicator for y is not supported.')
        X, y = self._validate_data(X, y, multi_output=True, accept_sparse='csc', dtype=DTYPE, force_all_finite=False)
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = estimator._compute_missing_values_in_feature_mask(X, estimator_name=self.__class__.__name__)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        if issparse(X):
            X.sort_indices()
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn('A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().', DataConversionWarning, stacklevel=2)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        if self.criterion == 'poisson':
            if np.any(y < 0):
                raise ValueError('Some value(s) of y are negative which is not allowed for Poisson regression.')
            if np.sum(y) <= 0:
                raise ValueError('Sum of y is not strictly positive which is necessary for Poisson regression.')
        self._n_samples, self.n_outputs_ = y.shape
        y, expanded_class_weight = self._validate_y_class_weight(y)
        if getattr(y, 'dtype', None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError('`max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`.')
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples=X.shape[0], max_samples=self.max_samples)
        else:
            n_samples_bootstrap = None
        self._n_samples_bootstrap = n_samples_bootstrap
        self._validate_estimator()
        if not self.bootstrap and self.oob_score:
            raise ValueError('Out of bag estimation only available if bootstrap=True')
        random_state = check_random_state(self.random_state)
        if not self.warm_start or not hasattr(self, 'estimators_'):
            self.estimators_ = []
        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to len(estimators_)=%d when warm_start==True' % (self.n_estimators, len(self.estimators_)))
        elif n_more_estimators == 0:
            warn('Warm-start fitting without increasing n_estimators does not fit new trees.')
        else:
            if self.warm_start and len(self.estimators_) > 0:
                random_state.randint(MAX_INT, size=len(self.estimators_))
            trees = [self._make_estimator(append=False, random_state=random_state) for i in range(n_more_estimators)]
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')((delayed(_parallel_build_trees)(t, self.bootstrap, X, y, sample_weight, i, len(trees), verbose=self.verbose, class_weight=self.class_weight, n_samples_bootstrap=n_samples_bootstrap, missing_values_in_feature_mask=missing_values_in_feature_mask) for i, t in enumerate(trees)))
            self.estimators_.extend(trees)
        if self.oob_score and (n_more_estimators > 0 or not hasattr(self, 'oob_score_')):
            y_type = type_of_target(y)
            if y_type == 'unknown' or (self._estimator_type == 'classifier' and y_type == 'multiclass-multioutput'):
                raise ValueError(f'The type of target cannot be used to compute OOB estimates. Got {y_type} while only the following are supported: continuous, continuous-multioutput, binary, multiclass, multilabel-indicator.')
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(X, y, scoring_function=self.oob_score)
            else:
                self._set_oob_score_and_attributes(X, y)
        if hasattr(self, 'classes_') and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
        return self

    @abstractmethod
    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Default depends on whether
            this is a regression (R2 score) or classification problem
            (accuracy score).
        """

    def _compute_oob_predictions(self, X, y):
        """Compute and set the OOB score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or                 (n_samples, 1, n_outputs)
            The OOB predictions.
        """
        if issparse(X):
            X = X.tocsr()
        n_samples = y.shape[0]
        n_outputs = self.n_outputs_
        if is_classifier(self) and hasattr(self, 'n_classes_'):
            oob_pred_shape = (n_samples, self.n_classes_[0], n_outputs)
        else:
            oob_pred_shape = (n_samples, 1, n_outputs)
        oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)
        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
            y_pred = self._get_oob_predictions(estimator, X[unsampled_indices, :])
            oob_pred[unsampled_indices, ...] += y_pred
            n_oob_pred[unsampled_indices, :] += 1
        for k in range(n_outputs):
            if (n_oob_pred == 0).any():
                warn('Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.', UserWarning)
                n_oob_pred[n_oob_pred == 0] = 1
            oob_pred[..., k] /= n_oob_pred[..., [k]]
        return oob_pred

    def _validate_y_class_weight(self, y):
        return (y, None)

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)
        if self.estimators_[0]._support_missing_values(X):
            force_all_finite = 'allow-nan'
        else:
            force_all_finite = True
        X = self._validate_data(X, dtype=DTYPE, accept_sparse='csr', reset=False, force_all_finite=force_all_finite)
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError('No support for np.int64 index based sparse matrices')
        return X

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

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
        check_is_fitted(self)
        all_importances = Parallel(n_jobs=self.n_jobs, prefer='threads')((delayed(getattr)(tree, 'feature_importances_') for tree in self.estimators_ if tree.tree_.node_count > 1))
        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)
        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def _get_estimators_indices(self):
        for tree in self.estimators_:
            if not self.bootstrap:
                yield np.arange(self._n_samples, dtype=np.int32)
            else:
                seed = tree.random_state
                yield _generate_sample_indices(seed, self._n_samples, self._n_samples_bootstrap)

    @property
    def estimators_samples_(self):
        """The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices for sample_indices in self._get_estimators_indices()]

    def _more_tags(self):
        estimator = type(self.estimator)(criterion=self.criterion)
        return {'allow_nan': _safe_tags(estimator, key='allow_nan')}