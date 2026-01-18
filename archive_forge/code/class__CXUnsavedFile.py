from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class _CXUnsavedFile(Structure):
    """Helper for passing unsaved file arguments."""
    _fields_ = [('name', c_char_p), ('contents', c_char_p), ('length', c_ulong)]