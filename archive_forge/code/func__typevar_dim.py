from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
def _typevar_dim(name):
    """Simple temporary type constructor for typevar as a dim"""
    return ct.TypeVar(name)