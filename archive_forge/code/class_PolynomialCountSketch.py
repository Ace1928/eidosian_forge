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
class PolynomialCountSketch(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Polynomial kernel approximation via Tensor Sketch.

    Implements Tensor Sketch, which approximates the feature map
    of the polynomial kernel::

        K(X, Y) = (gamma * <X, Y> + coef0)^degree

    by efficiently computing a Count Sketch of the outer product of a
    vector with itself using Fast Fourier Transforms (FFT). Read more in the
    :ref:`User Guide <polynomial_kernel_approx>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of the polynomial kernel whose feature map
        will be approximated.

    degree : int, default=2
        Degree of the polynomial kernel whose feature map
        will be approximated.

    coef0 : int, default=0
        Constant term of the polynomial kernel whose feature map
        will be approximated.

    n_components : int, default=100
        Dimensionality of the output feature space. Usually, `n_components`
        should be greater than the number of features in input samples in
        order to achieve good performance. The optimal score / run time
        balance is typically achieved around `n_components` = 10 * `n_features`,
        but this depends on the specific dataset being used.

    random_state : int, RandomState instance, default=None
        Determines random number generation for indexHash and bitHash
        initialization. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

    Attributes
    ----------
    indexHash_ : ndarray of shape (degree, n_features), dtype=int64
        Array of indexes in range [0, n_components) used to represent
        the 2-wise independent hash functions for Count Sketch computation.

    bitHash_ : ndarray of shape (degree, n_features), dtype=float32
        Array with random entries in {+1, -1}, used to represent
        the 2-wise independent hash functions for Count Sketch computation.

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
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    Examples
    --------
    >>> from sklearn.kernel_approximation import PolynomialCountSketch
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> ps = PolynomialCountSketch(degree=3, random_state=1)
    >>> X_features = ps.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_kernel_approximation_plot_scalable_poly_kernels.py`
    """
    _parameter_constraints: dict = {'gamma': [Interval(Real, 0, None, closed='left')], 'degree': [Interval(Integral, 1, None, closed='left')], 'coef0': [Interval(Real, None, None, closed='neither')], 'n_components': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}

    def __init__(self, *, gamma=1.0, degree=2, coef0=0, n_components=100, random_state=None):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X.

        Initializes the internal variables. The method needs no information
        about the distribution of data, so we only care about n_features in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse='csc')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        if self.coef0 != 0:
            n_features += 1
        self.indexHash_ = random_state.randint(0, high=self.n_components, size=(self.degree, n_features))
        self.bitHash_ = random_state.choice(a=[-1, 1], size=(self.degree, n_features))
        self._n_features_out = self.n_components
        return self

    def transform(self, X):
        """Generate the feature map approximation for X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse='csc', reset=False)
        X_gamma = np.sqrt(self.gamma) * X
        if sp.issparse(X_gamma) and self.coef0 != 0:
            X_gamma = sp.hstack([X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))], format='csc')
        elif not sp.issparse(X_gamma) and self.coef0 != 0:
            X_gamma = np.hstack([X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))])
        if X_gamma.shape[1] != self.indexHash_.shape[1]:
            raise ValueError('Number of features of test samples does not match that of training samples.')
        count_sketches = np.zeros((X_gamma.shape[0], self.degree, self.n_components))
        if sp.issparse(X_gamma):
            for j in range(X_gamma.shape[1]):
                for d in range(self.degree):
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    count_sketches[:, d, iHashIndex] += (iHashBit * X_gamma[:, [j]]).toarray().ravel()
        else:
            for j in range(X_gamma.shape[1]):
                for d in range(self.degree):
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    count_sketches[:, d, iHashIndex] += iHashBit * X_gamma[:, j]
        count_sketches_fft = fft(count_sketches, axis=2, overwrite_x=True)
        count_sketches_fft_prod = np.prod(count_sketches_fft, axis=1)
        data_sketch = np.real(ifft(count_sketches_fft_prod, overwrite_x=True))
        return data_sketch