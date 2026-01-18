import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_get_tuple_sort_field_decl(a0, a1, a2, _elems=Elementaries(_lib.Z3_get_tuple_sort_field_decl)):
    r = _elems.f(a0, a1, a2)
    _elems.Check(a0)
    return r