from django.contrib.gis.db.backends.base.adapter import WKTAdapter
from django.contrib.gis.geos import GeometryCollection, Polygon
from django.db.backends.oracle.oracledb_any import oracledb
@classmethod
def _fix_geometry_collection(cls, coll):
    """
        Fix polygon orientations in geometry collections as described in
        __init__().
        """
    coll = coll.clone()
    for i, geom in enumerate(coll):
        if isinstance(geom, Polygon):
            coll[i] = cls._fix_polygon(geom, clone=False)
    return coll