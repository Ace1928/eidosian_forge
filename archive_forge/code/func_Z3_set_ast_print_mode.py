import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_set_ast_print_mode(a0, a1, _elems=Elementaries(_lib.Z3_set_ast_print_mode)):
    _elems.f(a0, a1)
    _elems.Check(a0)