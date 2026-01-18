import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_mk_re_range(a0, a1, a2, _elems=Elementaries(_lib.Z3_mk_re_range)):
    r = _elems.f(a0, a1, a2)
    _elems.Check(a0)
    return r