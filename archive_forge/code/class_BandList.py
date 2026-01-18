from ctypes import byref, c_double, c_int, c_void_p
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.shortcuts import numpy
from django.utils.encoding import force_str
from .const import (
class BandList(list):

    def __init__(self, source):
        self.source = source
        super().__init__()

    def __iter__(self):
        for idx in range(1, len(self) + 1):
            yield GDALBand(self.source, idx)

    def __len__(self):
        return capi.get_ds_raster_count(self.source._ptr)

    def __getitem__(self, index):
        try:
            return GDALBand(self.source, index + 1)
        except GDALException:
            raise GDALException('Unable to get band index %d' % index)