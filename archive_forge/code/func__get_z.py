from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
def _get_z(self, index):
    return capi.cs_getz(self.ptr, index, byref(c_double()))