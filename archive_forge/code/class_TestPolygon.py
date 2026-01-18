import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
class TestPolygon:

    def test_linearring(self):
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        ring = LinearRing(coords)
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == ring.coords[-1]
        assert ring.is_ring is True

    def test_polygon(self):
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        polygon = Polygon(coords)
        assert len(polygon.exterior.coords) == 5
        assert isinstance(polygon.exterior, LinearRing)
        ring = polygon.exterior
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == (0.0, 0.0)
        assert ring.is_ring is True
        assert len(polygon.interiors) == 0
        data = polygon.wkb
        polygon = None
        ring = None
        polygon = load_wkb(data)
        ring = polygon.exterior
        assert len(ring.coords) == 5
        assert ring.coords[0] == ring.coords[4]
        assert ring.coords[0] == (0.0, 0.0)
        assert ring.is_ring is True
        polygon = None
        polygon = Polygon(coords, [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])
        assert len(polygon.exterior.coords) == 5
        assert len(polygon.interiors[0].coords) == 5
        with pytest.raises(IndexError):
            polygon.interiors[1]
        with pytest.raises(NotImplementedError):
            polygon.coords
        assert polygon.__geo_interface__ == {'type': 'Polygon', 'coordinates': (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)), ((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)))}

    def test_linearring_empty(self):
        r_null = LinearRing()
        assert r_null.wkt == 'LINEARRING EMPTY'
        assert r_null.length == 0.0

    def test_dimensions(self):
        coords = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        polygon = Polygon(coords)
        assert polygon._ndim == 3
        gi = polygon.__geo_interface__
        assert gi['coordinates'] == (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),)
        e = polygon.exterior
        assert e._ndim == 3
        gi = e.__geo_interface__
        assert gi['coordinates'] == ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_attribute_chains(self):
        p = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        assert list(p.boundary.coords) == [(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 0.0)]
        ec = list(Point(0.0, 0.0).buffer(1.0, 1).exterior.coords)
        assert isinstance(ec, list)
        p = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)], [[(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25)]])
        assert p.area == 0.75
        'Not so much testing the exact values here, which are the\n        responsibility of the geometry engine (GEOS), but that we can get\n        chain functions and properties using anonymous references.\n        '
        assert list(p.interiors[0].coords) == [(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25), (-0.25, 0.25)]
        xy = list(p.interiors[0].buffer(1).exterior.coords)[0]
        assert len(xy) == 2
        ec = list(p.buffer(1).boundary.coords)
        assert isinstance(ec, list)

    def test_empty_equality(self):
        point1 = Point(0, 0)
        polygon1 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        polygon2 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
        polygon_empty1 = Polygon()
        polygon_empty2 = Polygon()
        assert point1 != polygon1
        assert polygon_empty1 == polygon_empty2
        assert polygon1 != polygon_empty1
        assert polygon1 == polygon2
        assert polygon_empty1 is not None

    def test_from_bounds(self):
        xmin, ymin, xmax, ymax = (-180, -90, 180, 90)
        coords = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        assert Polygon(coords) == Polygon.from_bounds(xmin, ymin, xmax, ymax)

    def test_empty_polygon_exterior(self):
        p = Polygon()
        assert p.exterior == LinearRing()