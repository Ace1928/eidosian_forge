import warnings
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics import euclidean_distances, pairwise_distances_argmin
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import check_is_fitted
class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor in the range `[0.5, 1.0)` is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, default=200
        Maximum number of iterations.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : bool, default=True
        Make a copy of input data.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        Which affinity to use. At the moment 'precomputed' and
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.

    verbose : bool, default=False
        Whether to be verbose.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AgglomerativeClustering : Recursively merges the pair of
        clusters that minimally increases a given linkage distance.
    FeatureAgglomeration : Similar to AgglomerativeClustering,
        but recursively merges features instead of samples.
    KMeans : K-Means clustering.
    MiniBatchKMeans : Mini-Batch K-Means clustering.
    MeanShift : Mean shift clustering using a flat kernel.
    SpectralClustering : Apply clustering to a projection
        of the normalized Laplacian.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When ``fit`` does not converge, ``cluster_centers_`` is still populated
    however it may be degenerate. In such a case, proceed with caution.
    If ``fit`` does not converge and fails to produce any ``cluster_centers_``
    then ``predict`` will label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])
    """
    _parameter_constraints: dict = {'damping': [Interval(Real, 0.5, 1.0, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'convergence_iter': [Interval(Integral, 1, None, closed='left')], 'copy': ['boolean'], 'preference': ['array-like', Interval(Real, None, None, closed='neither'), None], 'affinity': [StrOptions({'euclidean', 'precomputed'})], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, *, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state

    def _more_tags(self):
        return {'pairwise': self.affinity == 'precomputed'}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or                 array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Returns the instance itself.
        """
        if self.affinity == 'precomputed':
            accept_sparse = False
        else:
            accept_sparse = 'csr'
        X = self._validate_data(X, accept_sparse=accept_sparse)
        if self.affinity == 'precomputed':
            self.affinity_matrix_ = X.copy() if self.copy else X
        else:
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        if self.affinity_matrix_.shape[0] != self.affinity_matrix_.shape[1]:
            raise ValueError(f'The matrix of similarities must be a square array. Got {self.affinity_matrix_.shape} instead.')
        if self.preference is None:
            preference = np.median(self.affinity_matrix_)
        else:
            preference = self.preference
        preference = np.asarray(preference)
        random_state = check_random_state(self.random_state)
        self.cluster_centers_indices_, self.labels_, self.n_iter_ = _affinity_propagation(self.affinity_matrix_, max_iter=self.max_iter, convergence_iter=self.convergence_iter, preference=preference, damping=self.damping, verbose=self.verbose, return_n_iter=True, random_state=random_state)
        if self.affinity != 'precomputed':
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, accept_sparse='csr')
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("Predict method is not supported when affinity='precomputed'.")
        if self.cluster_centers_.shape[0] > 0:
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn("This model does not have any cluster centers because affinity propagation did not converge. Labeling every sample as '-1'.", ConvergenceWarning)
            return np.array([-1] * X.shape[0])

    def fit_predict(self, X, y=None):
        """Fit clustering from features/affinity matrix; return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or                 array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)