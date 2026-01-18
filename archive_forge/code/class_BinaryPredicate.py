from ctypes import c_byte, c_char_p, c_double
from django.contrib.gis.geos.libgeos import GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import check_predicate
class BinaryPredicate(UnaryPredicate):
    """For GEOS binary predicate functions."""
    argtypes = [GEOM_PTR, GEOM_PTR]