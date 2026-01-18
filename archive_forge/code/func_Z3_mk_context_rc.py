import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_mk_context_rc(a0, _elems=Elementaries(_lib.Z3_mk_context_rc)):
    r = _elems.f(a0)
    return r