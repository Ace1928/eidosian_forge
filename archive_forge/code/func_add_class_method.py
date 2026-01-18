import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def add_class_method(self, method, name, encoding):
    imp = add_method(self.objc_metaclass, name, method, encoding)
    self._imp_table[name] = imp