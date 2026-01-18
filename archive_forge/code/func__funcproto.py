from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
def _funcproto(args, ret):
    """Simple temporary type constructor for funcproto"""
    return ct.Function(*chain(args, (ret,)))