import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
class RTreeIndex(rtree.index.Index):
    """A simple wrapper around rtree's RTree Index

        Parameters
        ----------
        geometry : np.array of Shapely geometries
            Geometries from which to build the spatial index.
        """

    def __init__(self, geometry):
        stream = ((i, item.bounds, None) for i, item in enumerate(geometry) if pd.notnull(item) and (not item.is_empty))
        try:
            super().__init__(stream)
        except RTreeError:
            super().__init__()
        self.geometries = geometry
        self._prepared_geometries = np.array([None] * self.geometries.size, dtype=object)

    @property
    @doc(BaseSpatialIndex.valid_query_predicates)
    def valid_query_predicates(self):
        return {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches', 'covered_by', 'covers', 'contains_properly'}

    @doc(BaseSpatialIndex.query)
    def query(self, geometry, predicate=None, sort=False):
        if predicate not in self.valid_query_predicates:
            raise ValueError('Got `predicate` = `{}`, `predicate` must be one of {}'.format(predicate, self.valid_query_predicates))
        if hasattr(geometry, '__array__') and (not isinstance(geometry, BaseGeometry)):
            tree_index = []
            input_geometry_index = []
            for i, geo in enumerate(geometry):
                res = self.query(geo, predicate=predicate, sort=sort)
                tree_index.extend(res)
                input_geometry_index.extend([i] * len(res))
            return np.vstack([input_geometry_index, tree_index])
        if geometry is None:
            return np.array([], dtype=np.intp)
        if not isinstance(geometry, BaseGeometry):
            raise TypeError('Got `geometry` of type `{}`, `geometry` must be '.format(type(geometry)) + 'a shapely geometry.')
        if geometry.is_empty:
            return np.array([], dtype=np.intp)
        bounds = geometry.bounds
        tree_idx = list(self.intersection(bounds))
        if not tree_idx:
            return np.array([], dtype=np.intp)
        if predicate == 'within':
            res = []
            for index_in_tree in tree_idx:
                if self._prepared_geometries[index_in_tree] is None:
                    self._prepared_geometries[index_in_tree] = prep(self.geometries[index_in_tree])
                if self._prepared_geometries[index_in_tree].contains(geometry):
                    res.append(index_in_tree)
            tree_idx = res
        elif predicate is not None:
            if predicate in ('contains', 'intersects', 'covered_by', 'covers', 'contains_properly'):
                geometry = prep(geometry)
            tree_idx = [index_in_tree for index_in_tree in tree_idx if getattr(geometry, predicate)(self.geometries[index_in_tree])]
        if sort:
            return np.sort(np.array(tree_idx, dtype=np.intp))
        return np.array(tree_idx, dtype=np.intp)

    @doc(BaseSpatialIndex.query_bulk)
    def query_bulk(self, geometry, predicate=None, sort=False):
        warnings.warn('The `query_bulk()` method is deprecated and will be removed in GeoPandas 1.0. You can use the `query()` method instead.', FutureWarning, stacklevel=2)
        return self.query(geometry, predicate=predicate, sort=sort)

    def nearest(self, coordinates, num_results=1, objects=False):
        """
            Returns the nearest object or objects to the given coordinates.

            Requires rtree, and passes parameters directly to
            :meth:`rtree.index.Index.nearest`.

            This behaviour is deprecated and will be updated to be consistent
            with the pygeos PyGEOSSTRTreeIndex in a future release.

            If longer-term compatibility is required, use
            :meth:`rtree.index.Index.nearest` directly instead.

            Examples
            --------
            >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(3), range(3)))
            >>> s
            0    POINT (0.00000 0.00000)
            1    POINT (1.00000 1.00000)
            2    POINT (2.00000 2.00000)
            dtype: geometry

            >>> list(s.sindex.nearest((0, 0)))  # doctest: +SKIP
            [0]

            >>> list(s.sindex.nearest((0.5, 0.5)))  # doctest: +SKIP
            [0, 1]

            >>> list(s.sindex.nearest((3, 3), num_results=2))  # doctest: +SKIP
            [2, 1]

            >>> list(super(type(s.sindex), s.sindex).nearest((0, 0),
            ... num_results=2))  # doctest: +SKIP
            [0, 1]

            Parameters
            ----------
            coordinates : sequence or array
                This may be an object that satisfies the numpy array protocol,
                providing the index’s dimension * 2 coordinate pairs
                representing the mink and maxk coordinates in each dimension
                defining the bounds of the query window.
            num_results : integer
                The number of results to return nearest to the given
                coordinates. If two index entries are equidistant, both are
                returned. This property means that num_results may return more
                items than specified
            objects : True / False / ‘raw’
                If True, the nearest method will return index objects that were
                pickled when they were stored with each index entry, as well as
                the id and bounds of the index entries. If ‘raw’, it will
                return the object as entered into the database without the
                rtree.index.Item wrapper.
            """
        warnings.warn('sindex.nearest using the rtree backend was not previously documented and this behavior is deprecated in favor of matching the function signature provided by the pygeos backend (see PyGEOSSTRTreeIndex.nearest for details). This behavior will be updated in a future release.', FutureWarning, stacklevel=2)
        return super().nearest(coordinates, num_results=num_results, objects=objects)

    @doc(BaseSpatialIndex.intersection)
    def intersection(self, coordinates):
        return super().intersection(coordinates, objects=False)

    @property
    @doc(BaseSpatialIndex.size)
    def size(self):
        if hasattr(self, '_size'):
            size = self._size
        else:
            size = len(self.leaves()[0][1])
            self._size = size
        return size

    @property
    @doc(BaseSpatialIndex.is_empty)
    def is_empty(self):
        return self.geometries.size == 0 or self.size == 0

    def __len__(self):
        return self.size