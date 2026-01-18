import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
class lazy_property(object):

    def __init__(self, function):
        self.fget = function

    def __get__(self, obj, cls):
        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value