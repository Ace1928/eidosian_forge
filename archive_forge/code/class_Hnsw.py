import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
class Hnsw:
    """
    Class for building, loading and working with Hierarchical Navigable Small World index.
    """

    def __init__(self):
        """
        Create object for working with HNSW.
        """
        self._index = None
        self._data = None

    def build(self, pool, distance, max_neighbors=None, search_neighborhood_size=None, num_exact_candidates=None, batch_size=None, upper_level_batch_size=None, level_size_decay=None, num_threads=None, verbose=False, report_progress=True, snapshot_file=None, snapshot_interval=None):
        """
        Build index with given options.

        Parameters
        ----------
        pool : Pool
            Pool of vectors for which index will be built.

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
            Snapshot is saved after building each level also.
        """
        params = {}
        not_params = ['not_params', 'self', 'params', '__class__', 'pool', 'distance']
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        with log_fixup():
            self._index._build(json.dumps(params))

    def _check_index(self):
        if self._index is None:
            raise HnswException('Index is not built and not loaded')

    def save(self, index_path):
        """
        Save index to file.

        Parameters
        ----------
        index_path : string
            Path to file for saving index.
        """
        self._check_index()
        self._index._save(index_path)

    def load(self, index_path, pool, distance):
        """
        Load index from file.

        Parameters
        ----------
        index_path : string
            Path to file for loading index.

        pool : Pool
            Pool of vectors for which index will be loaded.

        distance : EDistance
            Distance that should be used for finding nearest vectors.
        """
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        self._index._load(index_path)
        self._data = None

    def load_from_bytes(self, index_data, pool, distance):
        """
        Load index from bytes.

        Parameters
        ----------
        index_data : bytes
            Index binary data.

        pool : Pool
            Pool of vectors for which index will be loaded.

        distance : EDistance
            Distance that should be used for finding nearest vectors.
        """
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        self._index._load_from_bytes(index_data)
        self._data = index_data

    def get_nearest(self, query, top_size, search_neighborhood_size, distance_calc_limit=0):
        """
        Get approximate nearest neighbors for query from index.

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.

        top_size : int
            Required number of neighbors.

        search_neighborhood_size : int
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of search time.
            It should be equal or greater than top_size.

        distance_calc_limit : int (default=0)
            Limit of distance calculation.
            To guarantee satisfactory search time at the expense of quality.
            0 is equivalent to no limit.

        Returns
        -------
        neighbors : list of tuples (id, distance)
        """
        self._check_index()
        return self._index._get_nearest(query, top_size, search_neighborhood_size, distance_calc_limit)