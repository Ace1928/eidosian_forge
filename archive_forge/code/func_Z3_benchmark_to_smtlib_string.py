import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_benchmark_to_smtlib_string(a0, a1, a2, a3, a4, a5, a6, a7, _elems=Elementaries(_lib.Z3_benchmark_to_smtlib_string)):
    r = _elems.f(a0, _str_to_bytes(a1), _str_to_bytes(a2), _str_to_bytes(a3), _str_to_bytes(a4), a5, a6, a7)
    _elems.Check(a0)
    return _to_pystr(r)