import unittest
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon
class GeoInterfaceTestCase(unittest.TestCase):

    def test_geointerface(self):
        d = {'type': 'Point', 'coordinates': (0.0, 0.0)}
        geom = shape(d)
        assert geom.geom_type == 'Point'
        assert tuple(geom.coords) == ((0.0, 0.0),)
        geom = None
        thing = GeoThing({'type': 'Point', 'coordinates': (0.0, 0.0)})
        geom = shape(thing)
        assert geom.geom_type == 'Point'
        assert tuple(geom.coords) == ((0.0, 0.0),)
        geom = shape({'type': 'LineString', 'coordinates': ((-1.0, -1.0), (1.0, 1.0))})
        assert isinstance(geom, LineString)
        assert tuple(geom.coords) == ((-1.0, -1.0), (1.0, 1.0))
        geom = shape({'type': 'LinearRing', 'coordinates': ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0))})
        assert isinstance(geom, LinearRing)
        assert tuple(geom.coords) == ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0))
        geom = shape({'type': 'Polygon', 'coordinates': (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0)), ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1)))})
        assert isinstance(geom, Polygon)
        assert tuple(geom.exterior.coords) == ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0), (0.0, 0.0))
        assert len(geom.interiors) == 1
        geom = shape({'type': 'MultiPoint', 'coordinates': ((1.0, 2.0), (3.0, 4.0))})
        assert isinstance(geom, MultiPoint)
        assert len(geom.geoms) == 2
        geom = shape({'type': 'MultiLineString', 'coordinates': (((0.0, 0.0), (1.0, 2.0)),)})
        assert isinstance(geom, MultiLineString)
        assert len(geom.geoms) == 1
        geom = shape({'type': 'MultiPolygon', 'coordinates': [(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)), ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1)))]})
        assert isinstance(geom, MultiPolygon)
        assert len(geom.geoms) == 1