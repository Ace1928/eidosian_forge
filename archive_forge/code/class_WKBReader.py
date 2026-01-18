from django.contrib.gis.geos.geometry import GEOSGeometry
from django.contrib.gis.geos.prototypes.io import (
class WKBReader(_WKBReader):

    def read(self, wkb):
        """Return a GEOSGeometry for the given WKB buffer."""
        return GEOSGeometry(super().read(wkb))