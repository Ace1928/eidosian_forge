from ctypes import c_byte, c_char_p, c_double
from django.contrib.gis.geos.libgeos import GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import check_predicate
class UnaryPredicate(GEOSFuncFactory):
    """For GEOS unary predicate functions."""
    argtypes = [GEOM_PTR]
    restype = c_byte
    errcheck = staticmethod(check_predicate)