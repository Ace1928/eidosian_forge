import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
Computes the minimum bounding circle that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> minimum_bounding_circle(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    <POLYGON ((12.071 5, 11.935 3.621, 11.533 2.294, 10.879 1.07...>
    >>> minimum_bounding_circle(LineString([(1, 1), (10, 10)]))
    <POLYGON ((11.864 5.5, 11.742 4.258, 11.38 3.065, 10.791 1.9...>
    >>> minimum_bounding_circle(MultiPoint([(2, 2), (4, 2)]))
    <POLYGON ((4 2, 3.981 1.805, 3.924 1.617, 3.831 1.444, 3.707...>
    >>> minimum_bounding_circle(Point(0, 1))
    <POINT (0 1)>
    >>> minimum_bounding_circle(GeometryCollection([]))
    <POLYGON EMPTY>

    See also
    --------
    minimum_bounding_radius
    