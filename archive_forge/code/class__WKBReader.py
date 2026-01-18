import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
class _WKBReader(IOBase):
    _constructor = wkb_reader_create
    ptr_type = WKB_READ_PTR
    destructor = wkb_reader_destroy

    def read(self, wkb):
        """Return a _pointer_ to C GEOS Geometry object from the given WKB."""
        if isinstance(wkb, memoryview):
            wkb_s = bytes(wkb)
            return wkb_reader_read(self.ptr, wkb_s, len(wkb_s))
        elif isinstance(wkb, bytes):
            return wkb_reader_read_hex(self.ptr, wkb, len(wkb))
        elif isinstance(wkb, str):
            wkb_s = wkb.encode()
            return wkb_reader_read_hex(self.ptr, wkb_s, len(wkb_s))
        else:
            raise TypeError