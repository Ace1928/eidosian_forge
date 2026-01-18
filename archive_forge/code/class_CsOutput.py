from ctypes import POINTER, c_byte, c_double, c_int, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import GEOSException, last_arg_byref
class CsOutput(GEOSFuncFactory):
    restype = CS_PTR

    @staticmethod
    def errcheck(result, func, cargs):
        if not result:
            raise GEOSException('Error encountered checking Coordinate Sequence returned from GEOS C function "%s".' % func.__name__)
        return result