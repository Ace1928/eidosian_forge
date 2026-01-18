from ctypes import c_byte
from django.contrib.gis.geos.libgeos import GEOM_PTR, PREPGEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import check_predicate
class PreparedPredicate(GEOSFuncFactory):
    argtypes = [PREPGEOM_PTR, GEOM_PTR]
    restype = c_byte
    errcheck = staticmethod(check_predicate)