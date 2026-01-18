import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def _radius_neighbors_reduce_func(self, dist, start, radius, return_distance):
    """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        radius : float
            The radius considered when making the nearest neighbors search.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : list of ndarray of shape (n_samples_chunk,)
            Returned only if `return_distance=True`.

        neigh : list of ndarray of shape (n_samples_chunk,)
            The neighbors indices.
        """
    neigh_ind = [np.where(d <= radius)[0] for d in dist]
    if return_distance:
        if self.effective_metric_ == 'euclidean':
            dist = [np.sqrt(d[neigh_ind[i]]) for i, d in enumerate(dist)]
        else:
            dist = [d[neigh_ind[i]] for i, d in enumerate(dist)]
        results = (dist, neigh_ind)
    else:
        results = neigh_ind
    return results