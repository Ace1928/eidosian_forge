from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
def _struct(names, dshapes):
    """Simple temporary type constructor for struct"""
    return ct.Record(list(zip(names, dshapes)))