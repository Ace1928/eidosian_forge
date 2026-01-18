from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def _get_instantiation(self):
    if self._data is None:
        f, l, c, o = (c_object_p(), c_uint(), c_uint(), c_uint())
        conf.lib.clang_getInstantiationLocation(self, byref(f), byref(l), byref(c), byref(o))
        if f:
            f = File(f)
        else:
            f = None
        self._data = (f, int(l.value), int(c.value), int(o.value))
    return self._data