import json
import os
import sys
import uuid
from ctypes import (
from pathlib import Path
from django.contrib.gis.gdal.driver import Driver
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.band import BandList
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.gdal.raster.const import (
from django.contrib.gis.gdal.srs import SpatialReference, SRSException
from django.contrib.gis.geometry import json_regex
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
class TransformPoint(list):
    indices = {'origin': (0, 3), 'scale': (1, 5), 'skew': (2, 4)}

    def __init__(self, raster, prop):
        x = raster.geotransform[self.indices[prop][0]]
        y = raster.geotransform[self.indices[prop][1]]
        super().__init__([x, y])
        self._raster = raster
        self._prop = prop

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        gtf = self._raster.geotransform
        gtf[self.indices[self._prop][0]] = value
        self._raster.geotransform = gtf

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        gtf = self._raster.geotransform
        gtf[self.indices[self._prop][1]] = value
        self._raster.geotransform = gtf