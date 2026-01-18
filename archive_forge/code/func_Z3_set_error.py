import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_set_error(a0, a1, _elems=Elementaries(_lib.Z3_set_error)):
    _elems.f(a0, a1)
    _elems.Check(a0)