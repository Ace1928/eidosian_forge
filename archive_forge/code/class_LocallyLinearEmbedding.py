from numbers import Integral, Real
import numpy as np
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from ..base import (
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import stable_cumsum
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
class LocallyLinearEmbedding(ClassNamePrefixFeaturesOutMixin, TransformerMixin, _UnstableArchMixin, BaseEstimator):
    """Locally Linear Embedding.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.

    n_components : int, default=2
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        The solver used to compute the eigenvectors. The available options are:

        - `'auto'` : algorithm will attempt to choose the best method for input
          data.
        - `'arpack'` : use arnoldi iteration in shift-invert mode. For this
          method, M may be a dense matrix, sparse matrix, or general linear
          operator.
        - `'dense'`  : use standard dense matrix operations for the eigenvalue
          decomposition. For this method, M must be an array or matrix type.
          This method should be avoided for large problems.

        .. warning::
           ARPACK can be unstable for some problems.  It is best to try several
           random seeds in order to check results.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.

    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - `standard`: use the standard locally linear embedding algorithm. see
          reference [1]_
        - `hessian`: use the Hessian eigenmap method. This method requires
          ``n_neighbors > n_components * (1 + (n_components + 1) / 2``. see
          reference [2]_
        - `modified`: use the modified locally linear embedding algorithm.
          see reference [3]_
        - `ltsa`: use local tangent space alignment algorithm. see
          reference [4]_

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'},                           default='auto'
        Algorithm to use for nearest neighbors search, passed to
        :class:`~sklearn.neighbors.NearestNeighbors` instance.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls. See :term:`Glossary <random_state>`.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : array-like, shape [n_samples, n_components]
        Stores the embedding vectors

    reconstruction_error_ : float
        Reconstruction error associated with `embedding_`

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    nbrs_ : NearestNeighbors object
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    See Also
    --------
    SpectralEmbedding : Spectral embedding for non-linear dimensionality
        reduction.
    TSNE : Distributed Stochastic Neighbor Embedding.

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import LocallyLinearEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = LocallyLinearEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """
    _parameter_constraints: dict = {'n_neighbors': [Interval(Integral, 1, None, closed='left')], 'n_components': [Interval(Integral, 1, None, closed='left')], 'reg': [Interval(Real, 0, None, closed='left')], 'eigen_solver': [StrOptions({'auto', 'arpack', 'dense'})], 'tol': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'method': [StrOptions({'standard', 'hessian', 'modified', 'ltsa'})], 'hessian_tol': [Interval(Real, 0, None, closed='left')], 'modified_tol': [Interval(Real, 0, None, closed='left')], 'neighbors_algorithm': [StrOptions({'auto', 'brute', 'kd_tree', 'ball_tree'})], 'random_state': ['random_state'], 'n_jobs': [None, Integral]}

    def __init__(self, *, n_neighbors=5, n_components=2, reg=0.001, eigen_solver='auto', tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm='auto', random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.neighbors_algorithm, n_jobs=self.n_jobs)
        random_state = check_random_state(self.random_state)
        X = self._validate_data(X, dtype=float)
        self.nbrs_.fit(X)
        self.embedding_, self.reconstruction_error_ = locally_linear_embedding(X=self.nbrs_, n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, method=self.method, hessian_tol=self.hessian_tol, modified_tol=self.modified_tol, random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)
        self._n_features_out = self.embedding_.shape[1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the embedding vectors for data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted `LocallyLinearEmbedding` class instance.
        """
        self._fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """
        Transform new points into embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Returns the instance itself.

        Notes
        -----
        Because of scaling performed by this method, it is discouraged to use
        it together with methods that are not scale-invariant (like SVMs).
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        ind = self.nbrs_.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=False)
        weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new