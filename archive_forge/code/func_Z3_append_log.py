import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_append_log(a0, _elems=Elementaries(_lib.Z3_append_log)):
    _elems.f(_str_to_bytes(a0))