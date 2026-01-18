from ctypes import byref, c_char_p, c_int
from enum import IntEnum
from types import NoneType
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import SRSException
from django.contrib.gis.gdal.libgdal import GDAL_VERSION
from django.contrib.gis.gdal.prototypes import srs as capi
from django.utils.encoding import force_bytes, force_str
class CoordTransform(GDALBase):
    """The coordinate system transformation object."""
    destructor = capi.destroy_ct

    def __init__(self, source, target):
        """Initialize on a source and target SpatialReference objects."""
        if not isinstance(source, SpatialReference) or not isinstance(target, SpatialReference):
            raise TypeError('source and target must be of type SpatialReference')
        self.ptr = capi.new_ct(source._ptr, target._ptr)
        self._srs1_name = source.name
        self._srs2_name = target.name

    def __str__(self):
        return 'Transform from "%s" to "%s"' % (self._srs1_name, self._srs2_name)