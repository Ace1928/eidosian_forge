from ctypes import byref, c_double
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.envelope import Envelope, OGREnvelope
from django.contrib.gis.gdal.error import GDALException, SRSException
from django.contrib.gis.gdal.feature import Feature
from django.contrib.gis.gdal.field import OGRFieldTypes
from django.contrib.gis.gdal.geometries import OGRGeometry
from django.contrib.gis.gdal.geomtype import OGRGeomType
from django.contrib.gis.gdal.prototypes import ds as capi
from django.contrib.gis.gdal.prototypes import geom as geom_api
from django.contrib.gis.gdal.prototypes import srs as srs_api
from django.contrib.gis.gdal.srs import SpatialReference
from django.utils.encoding import force_bytes, force_str
def _set_spatial_filter(self, filter):
    if isinstance(filter, OGRGeometry):
        capi.set_spatial_filter(self.ptr, filter.ptr)
    elif isinstance(filter, (tuple, list)):
        if not len(filter) == 4:
            raise ValueError('Spatial filter list/tuple must have 4 elements.')
        xmin, ymin, xmax, ymax = map(c_double, filter)
        capi.set_spatial_filter_rect(self.ptr, xmin, ymin, xmax, ymax)
    elif filter is None:
        capi.set_spatial_filter(self.ptr, None)
    else:
        raise TypeError('Spatial filter must be either an OGRGeometry instance, a 4-tuple, or None.')