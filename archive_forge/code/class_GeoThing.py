import unittest
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon
class GeoThing:

    def __init__(self, d):
        self.__geo_interface__ = d