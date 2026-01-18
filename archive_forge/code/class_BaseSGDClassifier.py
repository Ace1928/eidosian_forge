import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
class BaseSGDClassifier(LinearClassifierMixin, BaseSGD, metaclass=ABCMeta):
    loss_functions = {'hinge': (Hinge, 1.0), 'squared_hinge': (SquaredHinge, 1.0), 'perceptron': (Hinge, 0.0), 'log_loss': (Log,), 'modified_huber': (ModifiedHuber,), 'squared_error': (SquaredLoss,), 'huber': (Huber, DEFAULT_EPSILON), 'epsilon_insensitive': (EpsilonInsensitive, DEFAULT_EPSILON), 'squared_epsilon_insensitive': (SquaredEpsilonInsensitive, DEFAULT_EPSILON)}
    _parameter_constraints: dict = {**BaseSGD._parameter_constraints, 'loss': [StrOptions(set(loss_functions))], 'early_stopping': ['boolean'], 'validation_fraction': [Interval(Real, 0, 1, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None], 'class_weight': [StrOptions({'balanced'}), dict, None]}

    @abstractmethod
    def __init__(self, loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, warm_start=warm_start, average=average)
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _partial_fit(self, X, y, alpha, C, loss, learning_rate, max_iter, classes, sample_weight, coef_init, intercept_init):
        first_call = not hasattr(self, 'classes_')
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False, reset=first_call)
        n_samples, n_features = X.shape
        _check_partial_fit_first_call(self, classes)
        n_classes = self.classes_.shape[0]
        self._expanded_class_weight = compute_class_weight(self.class_weight, classes=self.classes_, y=y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        if getattr(self, 'coef_', None) is None or coef_init is not None:
            self._allocate_parameter_mem(n_classes=n_classes, n_features=n_features, input_dtype=X.dtype, coef_init=coef_init, intercept_init=intercept_init)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError('Number of features %d does not match previous data %d.' % (n_features, self.coef_.shape[-1]))
        self._loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, 't_'):
            self.t_ = 1.0
        if n_classes > 2:
            self._fit_multiclass(X, y, alpha=alpha, C=C, learning_rate=learning_rate, sample_weight=sample_weight, max_iter=max_iter)
        elif n_classes == 2:
            self._fit_binary(X, y, alpha=alpha, C=C, learning_rate=learning_rate, sample_weight=sample_weight, max_iter=max_iter)
        else:
            raise ValueError('The number of classes has to be greater than one; got %d class' % n_classes)
        return self

    def _fit(self, X, y, alpha, C, loss, learning_rate, coef_init=None, intercept_init=None, sample_weight=None):
        if hasattr(self, 'classes_'):
            delattr(self, 'classes_')
        y = self._validate_data(y=y)
        classes = np.unique(y)
        if self.warm_start and hasattr(self, 'coef_'):
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None
        if self.average > 0:
            self._standard_coef = self.coef_
            self._standard_intercept = self.intercept_
            self._average_coef = None
            self._average_intercept = None
        self.t_ = 1.0
        self._partial_fit(X, y, alpha, C, loss, learning_rate, self.max_iter, classes, sample_weight, coef_init, intercept_init)
        if self.tol is not None and self.tol > -np.inf and (self.n_iter_ == self.max_iter):
            warnings.warn('Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.', ConvergenceWarning)
        return self

    def _fit_binary(self, X, y, alpha, C, sample_weight, learning_rate, max_iter):
        """Fit a binary classifier on X and y."""
        coef, intercept, n_iter_ = fit_binary(self, 1, X, y, alpha, C, learning_rate, max_iter, self._expanded_class_weight[1], self._expanded_class_weight[0], sample_weight, random_state=self.random_state)
        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self._average_coef.reshape(1, -1)
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef.reshape(1, -1)
                self._standard_intercept = np.atleast_1d(intercept)
                self.intercept_ = self._standard_intercept
        else:
            self.coef_ = coef.reshape(1, -1)
            self.intercept_ = np.atleast_1d(intercept)

    def _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, require='sharedmem')((delayed(fit_binary)(self, i, X, y, alpha, C, learning_rate, max_iter, self._expanded_class_weight[i], 1.0, sample_weight, validation_mask=validation_mask, random_state=seed) for i, seed in enumerate(seeds)))
        n_iter_ = 0.0
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)
        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_
        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self._average_coef
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef
                self._standard_intercept = np.atleast_1d(self.intercept_)
                self.intercept_ = self._standard_intercept

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence, early stopping, and
        learning rate adjustments should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.

        y : ndarray of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if not hasattr(self, 'classes_'):
            self._more_validate_params(for_partial_fit=True)
            if self.class_weight == 'balanced':
                raise ValueError("class_weight '{0}' is not supported for partial_fit. In order to use 'balanced' weights, use compute_class_weight('{0}', classes=classes, y=y). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions. Pass the resulting weights as the class_weight parameter.".format(self.class_weight))
        return self._partial_fit(X, y, alpha=self.alpha, C=1.0, loss=self.loss, learning_rate=self.learning_rate, max_iter=1, classes=classes, sample_weight=sample_weight, coef_init=None, intercept_init=None)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_classes, n_features), default=None
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (n_classes,), default=None
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._more_validate_params()
        return self._fit(X, y, alpha=self.alpha, C=1.0, loss=self.loss, learning_rate=self.learning_rate, coef_init=coef_init, intercept_init=intercept_init, sample_weight=sample_weight)