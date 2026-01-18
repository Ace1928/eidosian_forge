import itertools
import numbers
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn
import numpy as np
from ..base import ClassifierMixin, RegressorMixin, _fit_context
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_random_state, column_or_1d, indices_to_mask
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_sample_weight, check_is_fitted, has_fit_parameter
from ._base import BaseEnsemble, _partition_estimators
class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict']), None], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'max_samples': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'max_features': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')], 'bootstrap': ['boolean'], 'bootstrap_features': ['boolean'], 'oob_score': ['boolean'], 'warm_start': ['boolean'], 'n_jobs': [None, Integral], 'random_state': ['random_state'], 'verbose': ['verbose']}

    @abstractmethod
    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator=estimator, n_estimators=n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'], dtype=None, force_all_finite=False, multi_output=True)
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _parallel_args(self):
        return {}

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None, check_input=True):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.

        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)
        self._validate_estimator()
        if max_depth is not None:
            self.estimator_.max_depth = max_depth
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])
        if max_samples > X.shape[0]:
            raise ValueError('max_samples must be <= n_samples')
        self._max_samples = max_samples
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_in_)
        if max_features > self.n_features_in_:
            raise ValueError('max_features must be <= n_features')
        max_features = max(1, int(max_features))
        self._max_features = max_features
        if not self.bootstrap and self.oob_score:
            raise ValueError('Out of bag estimation only available if bootstrap=True')
        if self.warm_start and self.oob_score:
            raise ValueError('Out of bag estimate only available if warm_start=False')
        if hasattr(self, 'oob_score_') and self.warm_start:
            del self.oob_score_
        if not self.warm_start or not hasattr(self, 'estimators_'):
            self.estimators_ = []
            self.estimators_features_ = []
        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to len(estimators_)=%d when warm_start==True' % (self.n_estimators, len(self.estimators_)))
        elif n_more_estimators == 0:
            warn('Warm-start fitting without increasing n_estimators does not fit new trees.')
            return self
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators, self.n_jobs)
        total_n_estimators = sum(n_estimators)
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))
        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args())((delayed(_parallel_build_estimators)(n_estimators[i], self, X, y, sample_weight, seeds[starts[i]:starts[i + 1]], total_n_estimators, verbose=self.verbose, check_input=check_input) for i in range(n_jobs)))
        self.estimators_ += list(itertools.chain.from_iterable((t[0] for t in all_results)))
        self.estimators_features_ += list(itertools.chain.from_iterable((t[1] for t in all_results)))
        if self.oob_score:
            self._set_oob_score(X, y)
        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        return y

    def _get_estimators_indices(self):
        for seed in self._seeds:
            feature_indices, sample_indices = _generate_bagging_indices(seed, self.bootstrap_features, self.bootstrap, self.n_features_in_, self._n_samples, self._max_features, self._max_samples)
            yield (feature_indices, sample_indices)

    @property
    def estimators_samples_(self):
        """
        The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices for _, sample_indices in self._get_estimators_indices()]