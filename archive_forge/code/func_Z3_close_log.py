import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_close_log(_elems=Elementaries(_lib.Z3_close_log)):
    _elems.f()