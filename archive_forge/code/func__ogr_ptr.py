from ctypes import c_uint
from django.contrib.gis import gdal
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry
def _ogr_ptr(self):
    return gdal.geometries.Point._create_empty() if self.empty else super()._ogr_ptr()