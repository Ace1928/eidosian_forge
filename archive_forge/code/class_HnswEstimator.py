import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
class HnswEstimator:
    """
    Class for building, loading and working with Hierarchical Navigable Small World index with SciKit-Learn
    Estimator compatible interface.
    Mostly drop-in replacement for sklearn.neighbors.NearestNeighbors (except for some parameters)
    """

    def __init__(self, n_neighbors=5, distance=EDistance.DotProduct, max_neighbors=32, search_neighborhood_size=300, num_exact_candidates=100, batch_size=1000, upper_level_batch_size=40000, level_size_decay=None):
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use by default for kneighbors queries.


        distance : EDistance
            Distance that should be used for finding nearest vectors.

        max_neighbors : int (default=32)
            Maximum number of neighbors that every item can be connected with.

        search_neighborhood_size : int (default=300)
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of building time.

        num_exact_candidates : int (default=100)
            Number of nearest vectors to take from batch.
            Higher values improve search quality in expense of building time.

        batch_size : int (default=1000)
            Number of items that added to graph on each step of algorithm.

        upper_level_batch_size : int (default=40000)
            Batch size for building upper levels.

        level_size_decay : int (default=max_neighbors/2)
            Base of exponent for decaying level sizes.
        """
        for key, value in iteritems(locals()):
            if key not in ['self', '__class__']:
                setattr(self, key, value)

    def _check_index(self):
        if self._index is None:
            raise HnswException('Index is not built and not loaded')

    def fit(self, X, y=None, num_threads=None, verbose=False, report_progress=True, snapshot_file=None, snapshot_interval=600):
        """
        Fit the HNSW model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_values)

        y: None
            Added to be compatible with Estimator API

        num_threads : int (default=number of CPUs)
            Number of threads for building index.

        report_progress : bool (default=True)
            Print progress of building.

        verbose : bool (default=False)
            Print additional information about time of building.

        snapshot_file : string (default=None)
            Path for saving snapshots during the index building.

        snapshot_interval : int (default=600)
            Interval between saving snapshots (seconds).

        Returns
        -------
        model : HnswEstimator

        """
        self._index, self._index_data = _hnsw._init_index(X, self.distance)
        params = self._get_params(return_none=False)
        not_params = ['not_params', 'self', 'params', '__class__', 'X', 'y']
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        del params['distance']
        with log_fixup():
            self._index._build(json.dumps(params))
        return self

    def _get_params(self, return_none):
        params = {}
        for key, value in self.__dict__.items():
            if key[0] != '_' and (return_none or value is not None):
                params[key] = value
        return params

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return self._get_params(return_none=True)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            HnswEstimator parameters.

        Returns
        -------
        self : HnswEstimator instance
        """
        if not params:
            return self
        valid_params = self._get_params(return_none=True)
        for key, value in params.items():
            if key not in valid_params:
                raise HnswException('Invalid parameter %s for HnswEstimator. Check the list of available parameters with `get_params().keys()`.')
                setattr(self, key, value)
        return self

    @property
    def effective_metric_(self):
        """
        Returns
        -------
        Distance that should be used for finding nearest vectors.
        """
        return self.distance

    @property
    def n_samples_fit_(self):
        """
        Returns
        -------
        Number of samples in the fitted data.
        """
        self._check_index()
        return self._index_data.shape[0]

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, search_neighborhood_size=None, distance_calc_limit=0):
        """Finds the approximate K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features) or None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        search_neighborhood_size : int, default=None
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of search time.
            It should be equal or greater than top_size.
            If None set to n_neighbors * 2.

        distance_calc_limit : int (default=0)
            Limit of distance calculation.
            To guarantee satisfactory search time at the expense of quality.
            0 is equivalent to no limit.

        Returns
        -------
        neigh_dist :numpy.ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True
        neigh_ind : numpy.ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        self._check_index()
        if X is None:
            X = self._index_data
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if search_neighborhood_size is None:
            search_neighborhood_size = n_neighbors * 2
        return self._index._kneighbors(X, n_neighbors, return_distance, self.distance, search_neighborhood_size, distance_calc_limit)