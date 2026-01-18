import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_solver_check_assumptions(a0, a1, a2, a3, _elems=Elementaries(_lib.Z3_solver_check_assumptions)):
    r = _elems.f(a0, a1, a2, a3)
    _elems.Check(a0)
    return r