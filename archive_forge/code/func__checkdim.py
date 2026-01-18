from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
def _checkdim(self, dim):
    """Check the given dimension."""
    if dim < 0 or dim > 2:
        raise GEOSException('invalid ordinate dimension "%d"' % dim)