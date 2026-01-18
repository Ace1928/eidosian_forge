import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_solver_propagate_created(a0, a1, a2, _elems=Elementaries(_lib.Z3_solver_propagate_created)):
    _elems.f(a0, a1, a2)
    _elems.Check(a0)