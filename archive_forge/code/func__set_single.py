from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
def _set_single(self, index, value):
    self._cs.setOrdinate(index, 0, value)