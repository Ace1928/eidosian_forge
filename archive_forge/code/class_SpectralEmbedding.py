import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import (
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.fixes import parse_version, sp_version
class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    affinity : {'nearest_neighbors', 'rbf', 'precomputed',                 'precomputed_nearest_neighbors'} or callable,                 default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf' : construct the affinity matrix by computing a radial basis
           function (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, default=None
        Kernel coefficient for rbf kernel. If None, gamma will be set to
        1/n_features.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems.
        If None, then ``'arpack'`` is used.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.

        .. versionadded:: 1.2

    n_neighbors : int, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_neighbors_ : int
        Number of nearest neighbors effectively used.

    See Also
    --------
    Isomap : Non-linear dimensionality reduction through Isometric Mapping.

    References
    ----------

    - :doi:`A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      <10.1007/s11222-007-9033-z>`

    - `On Spectral Clustering: Analysis and an algorithm, 2001
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      <https://citeseerx.ist.psu.edu/doc_view/pid/796c5d6336fc52aa84db575fb821c78918b65f58>`_

    - :doi:`Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      <10.1109/34.868688>`

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import SpectralEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = SpectralEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'affinity': [StrOptions({'nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'}), callable], 'gamma': [Interval(Real, 0, None, closed='left'), None], 'random_state': ['random_state'], 'eigen_solver': [StrOptions({'arpack', 'lobpcg', 'amg'}), None], 'eigen_tol': [Interval(Real, 0, None, closed='left'), StrOptions({'auto'})], 'n_neighbors': [Interval(Integral, 1, None, closed='left'), None], 'n_jobs': [None, Integral]}

    def __init__(self, n_components=2, *, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, eigen_tol='auto', n_neighbors=None, n_jobs=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _more_tags(self):
        return {'pairwise': self.affinity in ['precomputed', 'precomputed_nearest_neighbors']}

    def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : array-like of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        affinity_matrix of shape (n_samples, n_samples)
        """
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        if self.affinity == 'precomputed_nearest_neighbors':
            estimator = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric='precomputed').fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode='connectivity')
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            return self.affinity_matrix_
        if self.affinity == 'nearest_neighbors':
            if sparse.issparse(X):
                warnings.warn('Nearest neighbors affinity currently does not support sparse input, falling back to rbf affinity')
                self.affinity = 'rbf'
            else:
                self.n_neighbors_ = self.n_neighbors if self.n_neighbors is not None else max(int(X.shape[0] / 10), 1)
                self.affinity_matrix_ = kneighbors_graph(X, self.n_neighbors_, include_self=True, n_jobs=self.n_jobs)
                self.affinity_matrix_ = 0.5 * (self.affinity_matrix_ + self.affinity_matrix_.T)
                return self.affinity_matrix_
        if self.affinity == 'rbf':
            self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix}, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse='csr', ensure_min_samples=2)
        random_state = check_random_state(self.random_state)
        affinity_matrix = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(affinity_matrix, n_components=self.n_components, eigen_solver=self.eigen_solver, eigen_tol=self.eigen_tol, random_state=random_state)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix} of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        self.fit(X)
        return self.embedding_