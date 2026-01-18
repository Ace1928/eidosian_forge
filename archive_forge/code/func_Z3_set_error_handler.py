import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_set_error_handler(ctx, hndlr, _elems=Elementaries(_lib.Z3_set_error_handler)):
    ceh = _error_handler_type(hndlr)
    _elems.f(ctx, ceh)
    _elems.Check(ctx)
    return ceh