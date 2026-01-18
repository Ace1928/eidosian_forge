from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def _to_python_string(x, *args):
    return x