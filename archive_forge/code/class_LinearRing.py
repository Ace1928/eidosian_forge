from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry, LinearGeometryMixin
from django.contrib.gis.geos.point import Point
from django.contrib.gis.shortcuts import numpy
class LinearRing(LineString):
    _minlength = 4
    _init_func = capi.create_linearring

    @property
    def is_counterclockwise(self):
        if self.empty:
            raise ValueError('Orientation of an empty LinearRing cannot be determined.')
        return self._cs.is_counterclockwise