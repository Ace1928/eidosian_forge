import warnings
from heapq import heapify, heappop, heappush, heappushpop
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from ..base import (
from ..metrics import DistanceMetric
from ..metrics._dist_metrics import METRIC_MAPPING64
from ..metrics.pairwise import _VALID_METRICS, paired_distances
from ..utils import check_array
from ..utils._fast_dict import IntFloatDict
from ..utils._param_validation import (
from ..utils.graph import _fix_connected_components
from ..utils.validation import check_memory
from . import _hierarchical_fast as _hierarchical  # type: ignore
from ._feature_agglomeration import AgglomerationTransform
class FeatureAgglomeration(ClassNamePrefixFeaturesOutMixin, AgglomerativeClustering, AgglomerationTransform):
    """Agglomerate features.

    Recursively merges pair of clusters of features.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

        .. deprecated:: 1.4
           `metric=None` is deprecated in 1.4 and will be removed in 1.6.
           Let `metric` be the default value (i.e. `"euclidean"`) instead.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each feature the neighboring
        features following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is `None`, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at `n_clusters`. This is useful
        to decrease computation time if the number of clusters is not small
        compared to the number of features. This option is useful only when
        specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of features. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - "ward" minimizes the variance of the clusters being merged.
        - "complete" or maximum linkage uses the maximum distances between
          all features of the two sets.
        - "average" uses the average of the distances of each feature of
          the two sets.
        - "single" uses the minimum of the distances between all features
          of the two sets.

    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument `axis=1`, and reduce it to an array of size [M].

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array-like of (n_features,)
        Cluster labels for each feature.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_features`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_features` is a non-leaf
        node and has children `children_[i - n_features]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_features + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    AgglomerativeClustering : Agglomerative clustering samples instead of
        features.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, cluster
    >>> digits = datasets.load_digits()
    >>> images = digits.images
    >>> X = np.reshape(images, (len(images), -1))
    >>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
    >>> agglo.fit(X)
    FeatureAgglomeration(n_clusters=32)
    >>> X_reduced = agglo.transform(X)
    >>> X_reduced.shape
    (1797, 32)
    """
    _parameter_constraints: dict = {'n_clusters': [Interval(Integral, 1, None, closed='left'), None], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable, Hidden(None)], 'memory': [str, HasMethods('cache'), None], 'connectivity': ['array-like', callable, None], 'compute_full_tree': [StrOptions({'auto'}), 'boolean'], 'linkage': [StrOptions(set(_TREE_BUILDERS.keys()))], 'pooling_func': [callable], 'distance_threshold': [Interval(Real, 0, None, closed='left'), None], 'compute_distances': ['boolean']}

    def __init__(self, n_clusters=2, *, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func=np.mean, distance_threshold=None, compute_distances=False):
        super().__init__(n_clusters=n_clusters, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, metric=metric, distance_threshold=distance_threshold, compute_distances=compute_distances)
        self.pooling_func = pooling_func

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the hierarchical clustering on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = self._validate_data(X, ensure_min_features=2)
        super()._fit(X.T)
        self._n_features_out = self.n_clusters_
        return self

    @property
    def fit_predict(self):
        """Fit and return the result of each sample's clustering assignment."""
        raise AttributeError