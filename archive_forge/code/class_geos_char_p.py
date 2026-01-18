from ctypes import POINTER, c_char_p, c_int, c_ubyte, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import (
class geos_char_p(c_char_p):
    pass