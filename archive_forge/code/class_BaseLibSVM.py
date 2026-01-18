import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for estimators that use libsvm as backing library.

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    """
    _parameter_constraints: dict = {'kernel': [StrOptions({'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}), callable], 'degree': [Interval(Integral, 0, None, closed='left')], 'gamma': [StrOptions({'scale', 'auto'}), Interval(Real, 0.0, None, closed='left')], 'coef0': [Interval(Real, None, None, closed='neither')], 'tol': [Interval(Real, 0.0, None, closed='neither')], 'C': [Interval(Real, 0.0, None, closed='neither')], 'nu': [Interval(Real, 0.0, 1.0, closed='right')], 'epsilon': [Interval(Real, 0.0, None, closed='left')], 'shrinking': ['boolean'], 'probability': ['boolean'], 'cache_size': [Interval(Real, 0, None, closed='neither')], 'class_weight': [StrOptions({'balanced'}), dict, None], 'verbose': ['verbose'], 'max_iter': [Interval(Integral, -1, None, closed='left')], 'random_state': ['random_state']}
    _sparse_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu, epsilon, shrinking, probability, cache_size, class_weight, verbose, max_iter, random_state):
        if self._impl not in LIBSVM_IMPL:
            raise ValueError('impl should be one of %s, %s was given' % (LIBSVM_IMPL, self._impl))
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    def _more_tags(self):
        return {'pairwise': self.kernel == 'precomputed'}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)                 or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        rnd = check_random_state(self.random_state)
        sparse = sp.issparse(X)
        if sparse and self.kernel == 'precomputed':
            raise TypeError('Sparse precomputed kernels are not supported.')
        self._sparse = sparse and (not callable(self.kernel))
        if callable(self.kernel):
            check_consistent_length(X, y)
        else:
            X, y = self._validate_data(X, y, dtype=np.float64, order='C', accept_sparse='csr', accept_large_sparse=False)
        y = self._validate_targets(y)
        sample_weight = np.asarray([] if sample_weight is None else sample_weight, dtype=np.float64)
        solver_type = LIBSVM_IMPL.index(self._impl)
        n_samples = _num_samples(X)
        if solver_type != 2 and n_samples != y.shape[0]:
            raise ValueError('X and y have incompatible shapes.\n' + 'X has %s samples, but y has %s.' % (n_samples, y.shape[0]))
        if self.kernel == 'precomputed' and n_samples != X.shape[1]:
            raise ValueError('Precomputed matrix must be a square matrix. Input is a {}x{} matrix.'.format(X.shape[0], X.shape[1]))
        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != n_samples:
            raise ValueError('sample_weight and X have incompatible shapes: %r vs %r\nNote: Sparse matrices cannot be indexed w/boolean masks (use `indices=True` in CV).' % (sample_weight.shape, X.shape))
        kernel = 'precomputed' if callable(self.kernel) else self.kernel
        if kernel == 'precomputed':
            self._gamma = 0.0
        elif isinstance(self.gamma, str):
            if self.gamma == 'scale':
                X_var = X.multiply(X).mean() - X.mean() ** 2 if sparse else X.var()
                self._gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
            elif self.gamma == 'auto':
                self._gamma = 1.0 / X.shape[1]
        elif isinstance(self.gamma, Real):
            self._gamma = self.gamma
        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:
            print('[LibSVM]', end='')
        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        self.shape_fit_ = X.shape if hasattr(X, 'shape') else (n_samples,)
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_
        dual_coef = self._dual_coef_.data if self._sparse else self._dual_coef_
        intercept_finiteness = np.isfinite(self._intercept_).all()
        dual_coef_finiteness = np.isfinite(dual_coef).all()
        if not (intercept_finiteness and dual_coef_finiteness):
            raise ValueError('The dual coefficients or intercepts are not finite. The input data may contain large values and need to be preprocessed.')
        if self._impl in ['c_svc', 'nu_svc']:
            self.n_iter_ = self._num_iter
        else:
            self.n_iter_ = self._num_iter.item()
        return self

    def _validate_targets(self, y):
        """Validation of y and class_weight.

        Default implementation for SVR and one-class; overridden in BaseSVC.
        """
        return column_or_1d(y, warn=True).astype(np.float64, copy=False)

    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).  Consider pre-processing your data with StandardScaler or MinMaxScaler.' % self.max_iter, ConvergenceWarning)

    def _dense_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        if callable(self.kernel):
            self.__Xfit = X
            X = self._compute_kernel(X)
            if X.shape[0] != X.shape[1]:
                raise ValueError('X.shape[0] should be equal to X.shape[1]')
        libsvm.set_verbosity_wrap(self.verbose)
        self.support_, self.support_vectors_, self._n_support, self.dual_coef_, self.intercept_, self._probA, self._probB, self.fit_status_, self._num_iter = libsvm.fit(X, y, svm_type=solver_type, sample_weight=sample_weight, class_weight=getattr(self, 'class_weight_', np.empty(0)), kernel=kernel, C=self.C, nu=self.nu, probability=self.probability, degree=self.degree, shrinking=self.shrinking, tol=self.tol, cache_size=self.cache_size, coef0=self.coef0, gamma=self._gamma, epsilon=self.epsilon, max_iter=self.max_iter, random_seed=random_seed)
        self._warn_from_fit_status()

    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        X.sort_indices()
        kernel_type = self._sparse_kernels.index(kernel)
        libsvm_sparse.set_verbosity_wrap(self.verbose)
        self.support_, self.support_vectors_, dual_coef_data, self.intercept_, self._n_support, self._probA, self._probB, self.fit_status_, self._num_iter = libsvm_sparse.libsvm_sparse_train(X.shape[1], X.data, X.indices, X.indptr, y, solver_type, kernel_type, self.degree, self._gamma, self.coef0, self.tol, self.C, getattr(self, 'class_weight_', np.empty(0)), sample_weight, self.nu, self.cache_size, self.epsilon, int(self.shrinking), int(self.probability), self.max_iter, random_seed)
        self._warn_from_fit_status()
        if hasattr(self, 'classes_'):
            n_class = len(self.classes_) - 1
        else:
            n_class = 1
        n_SV = self.support_vectors_.shape[0]
        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        if not n_SV:
            self.dual_coef_ = sp.csr_matrix([])
        else:
            dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1, dual_coef_indices.size / n_class)
            self.dual_coef_ = sp.csr_matrix((dual_coef_data, dual_coef_indices, dual_coef_indptr), (n_class, n_SV))

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):
        X = self._compute_kernel(X)
        if X.ndim == 1:
            X = check_array(X, order='C', accept_large_sparse=False)
        kernel = self.kernel
        if callable(self.kernel):
            kernel = 'precomputed'
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
        svm_type = LIBSVM_IMPL.index(self._impl)
        return libsvm.predict(X, self.support_, self.support_vectors_, self._n_support, self._dual_coef_, self._intercept_, self._probA, self._probB, svm_type=svm_type, kernel=kernel, degree=self.degree, coef0=self.coef0, gamma=self._gamma, cache_size=self.cache_size)

    def _sparse_predict(self, X):
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'
        kernel_type = self._sparse_kernels.index(kernel)
        C = 0.0
        return libsvm_sparse.libsvm_sparse_predict(X.data, X.indices, X.indptr, self.support_vectors_.data, self.support_vectors_.indices, self.support_vectors_.indptr, self._dual_coef_.data, self._intercept_, LIBSVM_IMPL.index(self._impl), kernel_type, self.degree, self._gamma, self.coef0, self.tol, C, getattr(self, 'class_weight_', np.empty(0)), self.nu, self.epsilon, self.shrinking, self.probability, self._n_support, self._probA, self._probB)

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X

    def _decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)
        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            return -dec_func.ravel()
        return dec_func

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order='C', accept_large_sparse=False)
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'
        return libsvm.decision_function(X, self.support_, self.support_vectors_, self._n_support, self._dual_coef_, self._intercept_, self._probA, self._probB, svm_type=LIBSVM_IMPL.index(self._impl), kernel=kernel, degree=self.degree, cache_size=self.cache_size, coef0=self.coef0, gamma=self._gamma)

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        kernel = self.kernel
        if hasattr(kernel, '__call__'):
            kernel = 'precomputed'
        kernel_type = self._sparse_kernels.index(kernel)
        return libsvm_sparse.libsvm_sparse_decision_function(X.data, X.indices, X.indptr, self.support_vectors_.data, self.support_vectors_.indices, self.support_vectors_.indptr, self._dual_coef_.data, self._intercept_, LIBSVM_IMPL.index(self._impl), kernel_type, self.degree, self._gamma, self.coef0, self.tol, self.C, getattr(self, 'class_weight_', np.empty(0)), self.nu, self.epsilon, self.shrinking, self.probability, self._n_support, self._probA, self._probB)

    def _validate_for_predict(self, X):
        check_is_fitted(self)
        if not callable(self.kernel):
            X = self._validate_data(X, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=False, reset=False)
        if self._sparse and (not sp.issparse(X)):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()
        if sp.issparse(X) and (not self._sparse) and (not callable(self.kernel)):
            raise ValueError('cannot use sparse input in %r trained on dense data' % type(self).__name__)
        if self.kernel == 'precomputed':
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
        sv = self.support_vectors_
        if not self._sparse and sv.size > 0 and (self.n_support_.sum() != sv.shape[0]):
            raise ValueError(f'The internal representation of {self.__class__.__name__} was altered')
        return X

    @property
    def coef_(self):
        """Weights assigned to the features when `kernel="linear"`.

        Returns
        -------
        ndarray of shape (n_features, n_classes)
        """
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a linear kernel')
        coef = self._get_coef()
        if sp.issparse(coef):
            coef.data.flags.writeable = False
        else:
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)

    @property
    def n_support_(self):
        """Number of support vectors for each class."""
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError
        svm_type = LIBSVM_IMPL.index(self._impl)
        if svm_type in (0, 1):
            return self._n_support
        else:
            return np.array([self._n_support[0]])