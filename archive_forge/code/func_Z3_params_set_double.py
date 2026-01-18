import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_params_set_double(a0, a1, a2, a3, _elems=Elementaries(_lib.Z3_params_set_double)):
    _elems.f(a0, a1, a2, a3)
    _elems.Check(a0)