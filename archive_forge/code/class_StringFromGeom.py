from ctypes import POINTER, c_char_p, c_int, c_ubyte, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import (
class StringFromGeom(GEOSFuncFactory):
    """Argument is a Geometry, return type is a string."""
    argtypes = [GEOM_PTR]
    restype = geos_char_p
    errcheck = staticmethod(check_string)