import warnings
from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase
The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        distances_X : ndarray of shape (n_queries, self.n_neighbors)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.

        neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        Returns
        -------
        local_reachability_density : ndarray of shape (n_queries,)
            The local reachability density of each sample.
        