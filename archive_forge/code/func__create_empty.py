from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
@classmethod
def _create_empty(cls):
    return cls._create_point(None, None)