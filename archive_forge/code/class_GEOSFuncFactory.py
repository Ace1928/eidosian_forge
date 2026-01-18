import logging
import os
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_char_p
from ctypes.util import find_library
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import SimpleLazyObject, cached_property
from django.utils.version import get_version_tuple
class GEOSFuncFactory:
    """
    Lazy loading of GEOS functions.
    """
    argtypes = None
    restype = None
    errcheck = None

    def __init__(self, func_name, *, restype=None, errcheck=None, argtypes=None):
        self.func_name = func_name
        if restype is not None:
            self.restype = restype
        if errcheck is not None:
            self.errcheck = errcheck
        if argtypes is not None:
            self.argtypes = argtypes

    def __call__(self, *args):
        return self.func(*args)

    @cached_property
    def func(self):
        from django.contrib.gis.geos.prototypes.threadsafe import GEOSFunc
        func = GEOSFunc(self.func_name)
        func.argtypes = self.argtypes or []
        func.restype = self.restype
        if self.errcheck:
            func.errcheck = self.errcheck
        return func