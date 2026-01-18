import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
class SparseRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through sparse random projection.

    Sparse random matrix is an alternative to dense random
    projection matrix that guarantees similar embedding quality while being
    much more memory efficient and allowing faster computation of the
    projected data.

    If we note `s = 1 / density` the components of the random matrix are
    drawn from:

      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    density : float or 'auto', default='auto'
        Ratio in the range (0, 1] of non-zero component in the random
        projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'. This value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    dense_output : bool, default=False
        If True, ensure that the output of the random projection is a
        dense numpy array even if the input and random projection matrix
        are both sparse. In practice, if the number of components is
        small the number of zero components in the projected data will
        be very small and it will be more CPU and memory efficient to
        use a dense representation.

        If False, the projected data uses a sparse representation if
        the input is sparse.

    compute_inverse_components : bool, default=False
        Learn the inverse transform by computing the pseudo-inverse of the
        components during fit. Note that the pseudo-inverse is always a dense
        array, even if the training data was sparse. This means that it might be
        necessary to call `inverse_transform` on a small batch of samples at a
        time to avoid exhausting the available memory on the host. Moreover,
        computing the pseudo-inverse does not scale well to large matrices.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : sparse matrix of shape (n_components, n_features)
        Random matrix used for the projection. Sparse matrix will be of CSR
        format.

    inverse_components_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of the components, only computed if
        `compute_inverse_components` is True.

        .. versionadded:: 1.1

    density_ : float in range 0.0 - 1.0
        Concrete density computed from when density = "auto".

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianRandomProjection : Reduce dimensionality through Gaussian
        random projection.

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           https://cgi.di.uoa.gr/~optas/papers/jl.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import SparseRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = SparseRandomProjection(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    >>> # very few components are non-zero
    >>> np.mean(transformer.components_ != 0)
    0.0182...
    """
    _parameter_constraints: dict = {**BaseRandomProjection._parameter_constraints, 'density': [Interval(Real, 0.0, 1.0, closed='right'), StrOptions({'auto'})], 'dense_output': ['boolean']}

    def __init__(self, n_components='auto', *, density='auto', eps=0.1, dense_output=False, compute_inverse_components=False, random_state=None):
        super().__init__(n_components=n_components, eps=eps, compute_inverse_components=compute_inverse_components, random_state=random_state)
        self.dense_output = dense_output
        self.density = density

    def _make_random_matrix(self, n_components, n_features):
        """Generate the random projection matrix

        Parameters
        ----------
        n_components : int
            Dimensionality of the target projection space.

        n_features : int
            Dimensionality of the original source space.

        Returns
        -------
        components : sparse matrix of shape (n_components, n_features)
            The generated random matrix in CSR format.

        """
        random_state = check_random_state(self.random_state)
        self.density_ = _check_density(self.density, n_features)
        return _sparse_random_matrix(n_components, n_features, density=self.density_, random_state=random_state)

    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Projected array. It is a sparse matrix only when the input is sparse and
            `dense_output = False`.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=['csr', 'csc'], reset=False, dtype=[np.float64, np.float32])
        return safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)