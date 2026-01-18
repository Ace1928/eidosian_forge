import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import (
from .metrics.pairwise import KERNEL_PARAMS, PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from .utils import check_random_state, deprecated
from .utils._param_validation import Interval, StrOptions
from .utils.extmath import safe_sparse_dot
from .utils.validation import (
class Nystroem(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.

    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as `kernel_params`, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    coef0 : float, default=None
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    degree : float, default=None
        Degree of the polynomial kernel. Ignored by other kernels.

    kernel_params : dict, default=None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    n_components : int, default=100
        Number of features to construct.
        How many data points will be used to construct the mapping.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the uniform sampling without
        replacement of `n_components` of the training data to construct the
        basis kernel.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the kernel matrix into `n_jobs` even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.24

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Subset of training points used to construct the feature map.

    component_indices_ : ndarray of shape (n_components)
        Indices of ``components_`` in the training set.

    normalization_ : ndarray of shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    PolynomialCountSketch : Polynomial kernel approximation via Tensor Sketch.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    References
    ----------
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001

    * T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
      "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
      Comparison",
      Advances in Neural Information Processing Systems 2012

    Examples
    --------
    >>> from sklearn import datasets, svm
    >>> from sklearn.kernel_approximation import Nystroem
    >>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
    >>> data = X / 16.
    >>> clf = svm.LinearSVC(dual="auto")
    >>> feature_map_nystroem = Nystroem(gamma=.2,
    ...                                 random_state=1,
    ...                                 n_components=300)
    >>> data_transformed = feature_map_nystroem.fit_transform(data)
    >>> clf.fit(data_transformed, y)
    LinearSVC(dual='auto')
    >>> clf.score(data_transformed, y)
    0.9987...
    """
    _parameter_constraints: dict = {'kernel': [StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS.keys()) | {'precomputed'}), callable], 'gamma': [Interval(Real, 0, None, closed='left'), None], 'coef0': [Interval(Real, None, None, closed='neither'), None], 'degree': [Interval(Real, 1, None, closed='left'), None], 'kernel_params': [dict, None], 'n_components': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state'], 'n_jobs': [Integral, None]}

    def __init__(self, kernel='rbf', *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None, n_jobs=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs),                 default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]
        if self.n_components > n_samples:
            n_components = n_samples
            warnings.warn('n_components > n_samples. This is not possible.\nn_components was set to n_samples, which results in inefficient evaluation of the full kernel.')
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]
        basis_kernel = pairwise_kernels(basis, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **self._get_kernel_params())
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = basis_inds
        self._n_features_out = n_components
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse='csr', reset=False)
        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **kernel_params)
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel) and self.kernel != 'precomputed':
            for param in KERNEL_PARAMS[self.kernel]:
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        elif self.gamma is not None or self.coef0 is not None or self.degree is not None:
            raise ValueError("Don't pass gamma, coef0 or degree to Nystroem if using a callable or precomputed kernel")
        return params

    def _more_tags(self):
        return {'_xfail_checks': {'check_transformer_preserve_dtypes': 'dtypes are preserved but not at a close enough precision'}, 'preserves_dtype': [np.float64, np.float32]}