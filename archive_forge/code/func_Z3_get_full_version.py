import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_get_full_version(_elems=Elementaries(_lib.Z3_get_full_version)):
    r = _elems.f()
    return _to_pystr(r)