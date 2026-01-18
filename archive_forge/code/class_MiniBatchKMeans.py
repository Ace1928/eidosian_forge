import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import (
from ._k_means_common import (
from ._k_means_elkan import (
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse
class MiniBatchKMeans(_BaseKMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape             (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : 'auto' or int, default="auto"
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'` in version.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[3.375, 3.  ],
           [0.75 , 0.5 ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.55102041, 2.48979592],
           [1.06896552, 1.        ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    """
    _parameter_constraints: dict = {**_BaseKMeans._parameter_constraints, 'batch_size': [Interval(Integral, 1, None, closed='left')], 'compute_labels': ['boolean'], 'max_no_improvement': [Interval(Integral, 0, None, closed='left'), None], 'init_size': [Interval(Integral, 1, None, closed='left'), None], 'reassignment_ratio': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init='auto', reassignment_ratio=0.01):
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, verbose=verbose, random_state=random_state, tol=tol, n_init=n_init)
        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    def _check_params_vs_input(self, X):
        super()._check_params_vs_input(X, default_n_init=3)
        self._batch_size = min(self.batch_size, X.shape[0])
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(f'init_size={self._init_size} should be larger than n_clusters={self.n_clusters}. Setting it to min(3*n_clusters, n_samples)', RuntimeWarning, stacklevel=2)
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])
        if self.reassignment_ratio < 0:
            raise ValueError(f'reassignment_ratio should be >= 0, got {self.reassignment_ratio} instead.')

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(f'MiniBatchKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can prevent it by setting batch_size >= {self._n_threads * CHUNK_SIZE} or by setting the environment variable OMP_NUM_THREADS={n_active_threads}')

    def _mini_batch_convergence(self, step, n_steps, n_samples, centers_squared_diff, batch_inertia):
        """Helper function to encapsulate the early stopping logic"""
        batch_inertia /= self._batch_size
        step = step + 1
        if step == 1:
            if self.verbose:
                print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}')
            return False
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha
        if self.verbose:
            print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}, ewa inertia: {self._ewa_inertia}')
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f'Converged (small centers change) at step {step}/{n_steps}')
            return True
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1
        if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement:
            if self.verbose:
                print(f'Converged (lack of improvement in inertia) at step {step}/{n_steps}')
            return True
        return False

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= 10 * self.n_clusters:
            self._n_since_last_reassign = 0
            return True
        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False)
        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)
        self._check_mkl_vcomp(X, self._batch_size)
        x_squared_norms = row_norms(X, squared=True)
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f'Init {init_idx + 1}/{self._n_init} with method {init}')
            cluster_centers = self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=random_state, init_size=self._init_size, sample_weight=sample_weight)
            _, inertia = _labels_inertia_threadpool_limit(X_valid, sample_weight_valid, cluster_centers, n_threads=self._n_threads)
            if self.verbose:
                print(f'Inertia for init {init_idx + 1}/{self._n_init}: {inertia}')
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia
        centers = init_centers
        centers_new = np.empty_like(centers)
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0
        self._n_since_last_reassign = 0
        n_steps = self.max_iter * n_samples // self._batch_size
        with threadpool_limits(limits=1, user_api='blas'):
            for i in range(n_steps):
                minibatch_indices = random_state.randint(0, n_samples, self._batch_size)
                batch_inertia = _mini_batch_step(X=X[minibatch_indices], sample_weight=sample_weight[minibatch_indices], centers=centers, centers_new=centers_new, weight_sums=self._counts, random_state=random_state, random_reassign=self._random_reassign(), reassignment_ratio=self.reassignment_ratio, verbose=self.verbose, n_threads=self._n_threads)
                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0
                centers, centers_new = (centers_new, centers)
                if self._mini_batch_convergence(i, n_steps, n_samples, centers_squared_diff, batch_inertia):
                    break
        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil((i + 1) * self._batch_size / n_samples))
        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads)
        else:
            self.inertia_ = self._ewa_inertia * n_samples
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        self : object
            Return updated estimator.
        """
        has_centers = hasattr(self, 'cluster_centers_')
        X = self._validate_data(X, accept_sparse='csr', dtype=[np.float64, np.float32], order='C', accept_large_sparse=False, reset=not has_centers)
        self._random_state = getattr(self, '_random_state', check_random_state(self.random_state))
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, 'n_steps_', 0)
        x_squared_norms = row_norms(X, squared=True)
        if not has_centers:
            self._check_params_vs_input(X)
            self._n_threads = _openmp_effective_n_threads()
            init = self.init
            if _is_arraylike_not_scalar(init):
                init = check_array(init, dtype=X.dtype, copy=True, order='C')
                self._validate_center_shape(X, init)
            self._check_mkl_vcomp(X, X.shape[0])
            self.cluster_centers_ = self._init_centroids(X, x_squared_norms=x_squared_norms, init=init, random_state=self._random_state, init_size=self._init_size, sample_weight=sample_weight)
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)
            self._n_since_last_reassign = 0
        with threadpool_limits(limits=1, user_api='blas'):
            _mini_batch_step(X, sample_weight=sample_weight, centers=self.cluster_centers_, centers_new=self.cluster_centers_, weight_sums=self._counts, random_state=self._random_state, random_reassign=self._random_reassign(), reassignment_ratio=self.reassignment_ratio, verbose=self.verbose, n_threads=self._n_threads)
        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads)
        self.n_steps_ += 1
        self._n_features_out = self.cluster_centers_.shape[0]
        return self