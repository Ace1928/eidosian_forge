import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
class _LibraryWrapper(object):
    """
    Base class for wrapping a shared library. Do not use directly.
    """
    _lib = None
    _api_prototypes = {}

    def __init__(self):
        super(_LibraryWrapper, self).__init__()
        self._load()

    @classmethod
    def _load(cls):
        if cls._lib is not None:
            return cls._lib
        cls._lib = cls._cdll()
        for name, attrs in cls._api_prototypes.items():
            func = getattr(cls._lib, name)
            func.argtypes = attrs['argtypes']
            func.restype = attrs['restype']
            prefix = len('libevdev')
            pyname = dict.get(attrs, 'name', name[prefix:])
            setattr(cls, pyname, func)
        return cls._lib

    @staticmethod
    def _cdll():
        """Override in subclass"""
        raise NotImplementedError