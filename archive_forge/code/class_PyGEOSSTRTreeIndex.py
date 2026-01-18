import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
class PyGEOSSTRTreeIndex(BaseSpatialIndex):
    """A simple wrapper around pygeos's STRTree.


        Parameters
        ----------
        geometry : np.array of PyGEOS geometries
            Geometries from which to build the spatial index.
        """

    def __init__(self, geometry):
        non_empty = geometry.copy()
        non_empty[mod.is_empty(non_empty)] = None
        self._tree = mod.STRtree(non_empty)
        self.geometries = geometry.copy()

    @property
    def valid_query_predicates(self):
        """Returns valid predicates for the used spatial index.

            Returns
            -------
            set
                Set of valid predicates for this spatial index.

            Examples
            --------
            >>> from shapely.geometry import Point
            >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
            >>> s.sindex.valid_query_predicates  # doctest: +SKIP
            {None, "contains", "contains_properly", "covered_by", "covers", "crosses", "intersects", "overlaps", "touches", "within"}
            """
        return _PYGEOS_PREDICATES

    @doc(BaseSpatialIndex.query)
    def query(self, geometry, predicate=None, sort=False):
        if predicate not in self.valid_query_predicates:
            raise ValueError('Got `predicate` = `{}`; '.format(predicate) + '`predicate` must be one of {}'.format(self.valid_query_predicates))
        geometry = self._as_geometry_array(geometry)
        if compat.USE_SHAPELY_20:
            indices = self._tree.query(geometry, predicate=predicate)
        elif isinstance(geometry, np.ndarray):
            indices = self._tree.query_bulk(geometry, predicate=predicate)
        else:
            indices = self._tree.query(geometry, predicate=predicate)
        if sort:
            if indices.ndim == 1:
                return np.sort(indices)
            else:
                geo_idx, tree_idx = indices
                sort_indexer = np.lexsort((tree_idx, geo_idx))
                return np.vstack((geo_idx[sort_indexer], tree_idx[sort_indexer]))
        return indices

    @staticmethod
    def _as_geometry_array(geometry):
        """Convert geometry into a numpy array of PyGEOS geometries.

            Parameters
            ----------
            geometry
                An array-like of PyGEOS geometries, a GeoPandas GeoSeries/GeometryArray,
                shapely.geometry or list of shapely geometries.

            Returns
            -------
            np.ndarray
                A numpy array of pygeos geometries.
            """
        if isinstance(geometry, mod.Geometry):
            geometry = array._geom_to_shapely(geometry)
        if isinstance(geometry, np.ndarray):
            return array.from_shapely(geometry)._data
        elif isinstance(geometry, geoseries.GeoSeries):
            return geometry.values._data
        elif isinstance(geometry, array.GeometryArray):
            return geometry._data
        elif isinstance(geometry, BaseGeometry):
            return array._shapely_to_geom(geometry)
        elif geometry is None:
            return None
        elif isinstance(geometry, list):
            return np.asarray([array._shapely_to_geom(el) if isinstance(el, BaseGeometry) else el for el in geometry])
        else:
            return np.asarray(geometry)

    @doc(BaseSpatialIndex.query_bulk)
    def query_bulk(self, geometry, predicate=None, sort=False):
        warnings.warn('The `query_bulk()` method is deprecated and will be removed in GeoPandas 1.0. You can use the `query()` method instead.', FutureWarning, stacklevel=2)
        return self.query(geometry, predicate=predicate, sort=sort)

    @doc(BaseSpatialIndex.nearest)
    def nearest(self, geometry, return_all=True, max_distance=None, return_distance=False, exclusive=False):
        if not (compat.USE_SHAPELY_20 or compat.PYGEOS_GE_010):
            raise NotImplementedError('sindex.nearest requires shapely >= 2.0 or pygeos >= 0.10')
        if exclusive and (not compat.USE_SHAPELY_20):
            raise NotImplementedError('sindex.nearest exclusive parameter requires shapely >= 2.0')
        geometry = self._as_geometry_array(geometry)
        if isinstance(geometry, BaseGeometry) or geometry is None:
            geometry = [geometry]
        if compat.USE_SHAPELY_20:
            result = self._tree.query_nearest(geometry, max_distance=max_distance, return_distance=return_distance, all_matches=return_all, exclusive=exclusive)
        else:
            if not return_all and max_distance is None and (not return_distance):
                return self._tree.nearest(geometry)
            result = self._tree.nearest_all(geometry, max_distance=max_distance, return_distance=return_distance)
        if return_distance:
            indices, distances = result
        else:
            indices = result
        if not return_all and (not compat.USE_SHAPELY_20):
            mask = np.diff(indices[0, :]).astype('bool')
            mask = np.insert(mask, 0, True)
            indices = indices[:, mask]
            if return_distance:
                distances = distances[mask]
        if return_distance:
            return (indices, distances)
        else:
            return indices

    @doc(BaseSpatialIndex.intersection)
    def intersection(self, coordinates):
        try:
            iter(coordinates)
        except TypeError:
            raise TypeError('Invalid coordinates, must be iterable in format (minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). Got `coordinates` = {}.'.format(coordinates))
        if len(coordinates) == 4:
            indexes = self._tree.query(mod.box(*coordinates))
        elif len(coordinates) == 2:
            indexes = self._tree.query(mod.points(*coordinates))
        else:
            raise TypeError('Invalid coordinates, must be iterable in format (minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). Got `coordinates` = {}.'.format(coordinates))
        return indexes

    @property
    @doc(BaseSpatialIndex.size)
    def size(self):
        return len(self._tree)

    @property
    @doc(BaseSpatialIndex.is_empty)
    def is_empty(self):
        return len(self._tree) == 0

    def __len__(self):
        return len(self._tree)