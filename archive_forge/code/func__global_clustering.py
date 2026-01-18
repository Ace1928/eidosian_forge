import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Interval
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted
from . import AgglomerativeClustering
def _global_clustering(self, X=None):
    """
        Global clustering for the subclusters obtained after fitting
        """
    clusterer = self.n_clusters
    centroids = self.subcluster_centers_
    compute_labels = X is not None and self.compute_labels
    not_enough_centroids = False
    if isinstance(clusterer, Integral):
        clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        if len(centroids) < self.n_clusters:
            not_enough_centroids = True
    self._subcluster_norms = row_norms(self.subcluster_centers_, squared=True)
    if clusterer is None or not_enough_centroids:
        self.subcluster_labels_ = np.arange(len(centroids))
        if not_enough_centroids:
            warnings.warn('Number of subclusters found (%d) by BIRCH is less than (%d). Decrease the threshold.' % (len(centroids), self.n_clusters), ConvergenceWarning)
    else:
        self.subcluster_labels_ = clusterer.fit_predict(self.subcluster_centers_)
    if compute_labels:
        self.labels_ = self._predict(X)