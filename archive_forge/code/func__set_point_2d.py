from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
def _set_point_2d(self, index, value):
    x, y = value
    self._set_x(index, x)
    self._set_y(index, y)