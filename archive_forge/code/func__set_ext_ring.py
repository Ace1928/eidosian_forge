from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.geometry import GEOSGeometry
from django.contrib.gis.geos.libgeos import GEOM_PTR
from django.contrib.gis.geos.linestring import LinearRing
def _set_ext_ring(self, ring):
    """Set the exterior ring of the Polygon."""
    self[0] = ring