import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class _MXClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, clas=None):
        if clas is None:
            clas = type(obj)
        return self.fget.__get__(obj, clas)()

    def __set__(self, obj, value):
        if not self.fset:
            raise MXNetError('cannot use the setter: %s to set attribute' % obj.__name__)
        if inspect.isclass(obj):
            type_ = obj
            obj = None
        else:
            type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self