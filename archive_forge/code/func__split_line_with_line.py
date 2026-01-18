from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
@staticmethod
def _split_line_with_line(line, splitter):
    """Split a LineString with another (Multi)LineString or (Multi)Polygon"""
    if splitter.geom_type in ('Polygon', 'MultiPolygon'):
        splitter = splitter.boundary
    if not isinstance(line, LineString):
        raise GeometryTypeError('First argument must be a LineString')
    if not isinstance(splitter, LineString) and (not isinstance(splitter, MultiLineString)):
        raise GeometryTypeError('Second argument must be either a LineString or a MultiLineString')
    relation = splitter.relate(line)
    if relation[0] == '1':
        raise ValueError('Input geometry segment overlaps with the splitter.')
    elif relation[0] == '0' or relation[3] == '0':
        return line.difference(splitter)
    else:
        return [line]