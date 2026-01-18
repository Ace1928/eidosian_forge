import unittest
import pytest
from shapely import geometry
from shapely.ops import transform
class LambdaTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_point(self):
        g = geometry.Point(0, 1)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == 'Point'
        assert list(h.coords) == [(1.0, 2.0)]

    def test_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == 'LineString'
        assert list(h.coords) == [(1.0, 2.0), (3.0, 4.0)]

    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == 'Polygon'
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(2.0)

    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == 'MultiPolygon'
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(3.5)