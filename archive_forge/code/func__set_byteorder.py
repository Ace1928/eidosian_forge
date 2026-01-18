import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def _set_byteorder(self, order):
    if order not in (0, 1):
        raise ValueError('Byte order parameter must be 0 (Big Endian) or 1 (Little Endian).')
    wkb_writer_set_byteorder(self.ptr, order)