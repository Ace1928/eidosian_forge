from ctypes import POINTER, c_char_p, c_int, c_ubyte, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import (
class IntFromGeom(GEOSFuncFactory):
    """Argument is a geometry, return type is an integer."""
    argtypes = [GEOM_PTR]
    restype = c_int
    errcheck = staticmethod(check_minus_one)