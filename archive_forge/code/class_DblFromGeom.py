from ctypes import POINTER, c_double, c_int
from django.contrib.gis.geos.libgeos import GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import check_dbl, check_string
from django.contrib.gis.geos.prototypes.geom import geos_char_p
class DblFromGeom(GEOSFuncFactory):
    """
    Argument is a Geometry, return type is double that is passed
    in by reference as the last argument.
    """
    restype = c_int
    errcheck = staticmethod(check_dbl)