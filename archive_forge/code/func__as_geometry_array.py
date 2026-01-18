import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
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