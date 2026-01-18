from django.contrib.gis.db.backends.base.adapter import WKTAdapter
from django.contrib.gis.geos import GeometryCollection, Polygon
from django.db.backends.oracle.oracledb_any import oracledb
@staticmethod
def _polygon_must_be_fixed(poly):
    return not poly.empty and (not poly.exterior_ring.is_counterclockwise or any((x.is_counterclockwise for x in poly)))