import numpy as np
from shapely.errors import GeometryTypeError
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import LinearRing, Polygon
def _is_coordinates_empty(coordinates):
    """Helper to identify if coordinates or subset of coordinates are empty"""
    if coordinates is None:
        return True
    if isinstance(coordinates, (list, tuple, np.ndarray)):
        if len(coordinates) == 0:
            return True
        return all(map(_is_coordinates_empty, coordinates))
    else:
        return False