from warnings import warn
import shapely
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep
class SplitOp:

    @staticmethod
    def _split_polygon_with_line(poly, splitter):
        """Split a Polygon with a LineString"""
        if not isinstance(poly, Polygon):
            raise GeometryTypeError('First argument must be a Polygon')
        if not isinstance(splitter, LineString):
            raise GeometryTypeError('Second argument must be a LineString')
        union = poly.boundary.union(splitter)
        poly = prep(poly)
        return [pg for pg in polygonize(union) if poly.contains(pg.representative_point())]

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

    @staticmethod
    def _split_line_with_point(line, splitter):
        """Split a LineString with a Point"""
        if not isinstance(line, LineString):
            raise GeometryTypeError('First argument must be a LineString')
        if not isinstance(splitter, Point):
            raise GeometryTypeError('Second argument must be a Point')
        if not line.relate_pattern(splitter, '0********'):
            return [line]
        elif line.coords[0] == splitter.coords[0]:
            return [line]
        distance_on_line = line.project(splitter)
        coords = list(line.coords)
        current_position = 0.0
        for i in range(len(coords) - 1):
            point1 = coords[i]
            point2 = coords[i + 1]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            segment_length = (dx ** 2 + dy ** 2) ** 0.5
            current_position += segment_length
            if distance_on_line == current_position:
                return [LineString(coords[:i + 2]), LineString(coords[i + 1:])]
            elif distance_on_line < current_position:
                return [LineString(coords[:i + 1] + [splitter.coords[0]]), LineString([splitter.coords[0]] + coords[i + 1:])]
        return [line]

    @staticmethod
    def _split_line_with_multipoint(line, splitter):
        """Split a LineString with a MultiPoint"""
        if not isinstance(line, LineString):
            raise GeometryTypeError('First argument must be a LineString')
        if not isinstance(splitter, MultiPoint):
            raise GeometryTypeError('Second argument must be a MultiPoint')
        chunks = [line]
        for pt in splitter.geoms:
            new_chunks = []
            for chunk in filter(lambda x: not x.is_empty, chunks):
                new_chunks.extend(SplitOp._split_line_with_point(chunk, pt))
            chunks = new_chunks
        return chunks

    @staticmethod
    def split(geom, splitter):
        """
        Splits a geometry by another geometry and returns a collection of geometries. This function is the theoretical
        opposite of the union of the split geometry parts. If the splitter does not split the geometry, a collection
        with a single geometry equal to the input geometry is returned.
        The function supports:
          - Splitting a (Multi)LineString by a (Multi)Point or (Multi)LineString or (Multi)Polygon
          - Splitting a (Multi)Polygon by a LineString

        It may be convenient to snap the splitter with low tolerance to the geometry. For example in the case
        of splitting a line by a point, the point must be exactly on the line, for the line to be correctly split.
        When splitting a line by a polygon, the boundary of the polygon is used for the operation.
        When splitting a line by another line, a ValueError is raised if the two overlap at some segment.

        Parameters
        ----------
        geom : geometry
            The geometry to be split
        splitter : geometry
            The geometry that will split the input geom

        Example
        -------
        >>> pt = Point((1, 1))
        >>> line = LineString([(0,0), (2,2)])
        >>> result = split(line, pt)
        >>> result.wkt
        'GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1), LINESTRING (1 1, 2 2))'
        """
        if geom.geom_type in ('MultiLineString', 'MultiPolygon'):
            return GeometryCollection([i for part in geom.geoms for i in SplitOp.split(part, splitter).geoms])
        elif geom.geom_type == 'LineString':
            if splitter.geom_type in ('LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'):
                split_func = SplitOp._split_line_with_line
            elif splitter.geom_type == 'Point':
                split_func = SplitOp._split_line_with_point
            elif splitter.geom_type == 'MultiPoint':
                split_func = SplitOp._split_line_with_multipoint
            else:
                raise GeometryTypeError(f'Splitting a LineString with a {splitter.geom_type} is not supported')
        elif geom.geom_type == 'Polygon':
            if splitter.geom_type == 'LineString':
                split_func = SplitOp._split_polygon_with_line
            else:
                raise GeometryTypeError(f'Splitting a Polygon with a {splitter.geom_type} is not supported')
        else:
            raise GeometryTypeError(f'Splitting {geom.geom_type} geometry is not supported')
        return GeometryCollection(split_func(geom, splitter))