import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
class OnlineHnsw:
    """
    Class for building and working with Online Hierarchical Navigable Small World index.
    """

    def __init__(self, dtype, dimension, distance, max_neighbors=None, search_neighborhood_size=None, num_vertices=None, level_size_decay=None):
        """
        Create object with given options.

        Parameters
        ----------
        dtype : EVectorComponentType
            Type of vectors.
        dimension : int
            Dimension of vectors.
        distance : EDistance
            Distance that should be used for finding nearest vectors.
        max_neighbors : int (default=32)
            Maximum number of neighbors that every item can be connected with.
        search_neighborhood_size : int (default=300)
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of building time.
        num_vertices : int (default=0)
            Expected number of vectors in storage.
        level_size_decay : int (default=max_neighbors/2)
            Base of exponent for decaying level sizes.
        """
        self.dtype = dtype
        self.dimension = dimension
        params = {}
        all_params = ['max_neighbors', 'search_neighborhood_size', 'num_vertices', 'level_size_decay']
        for key, value in iteritems(locals()):
            if key in all_params and value is not None:
                params[key] = value
        self._online_index = _OnlineHnswDenseVectorIndex[dtype](dimension, distance, json.dumps(params))

    def get_nearest_and_add_item(self, query):
        """
        Get approximate nearest neighbors for query from index and add item to index

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
            Vector which should be added in index.

        Returns
        -------
        neighbors : list of tuples (id, distance) with length = search_neighborhood_size
        """
        return self._online_index._get_nearest_neighbors_and_add_item(query)

    def get_nearest(self, query, top_size=0):
        """
        Get approximate nearest neighbors for query from index.

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
        top_size : int
            Required number of neighbors.

        Returns
        -------
        neighbors : list of tuples (id, distance)
        """
        return self._online_index._get_nearest_neighbors(query, top_size)

    def add_item(self, item):
        """
        Add item in index.

        Parameters
        ----------
        item : list or numpy.ndarray
            Vector which should be added in index.
        """
        self._online_index._add_item(item)

    def get_item(self, id):
        """
        Get item from storage by id.

        Parameters
        ----------
        id : int
            Index of item in storage.

        Returns
        -------
        item : numpy.ndarray
        """
        return self._online_index._get_item(id)

    def get_num_items(self):
        """
        Get the number of items in storage.

        Returns
        -------
        num_items : int
        """
        return self._online_index._get_num_items()