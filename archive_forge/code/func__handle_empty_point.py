import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def _handle_empty_point(self, geom):
    from django.contrib.gis.geos import Point
    if isinstance(geom, Point) and geom.empty:
        if self.srid:
            geom = Point(float('NaN'), float('NaN'), srid=geom.srid)
        else:
            raise ValueError('Empty point is not representable in WKB.')
    return geom