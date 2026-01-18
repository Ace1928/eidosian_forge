import threading
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import CONTEXT_PTR, error_h, lgeos, notice_h
def _set_restype(self, restype):
    self.cfunc.restype = restype