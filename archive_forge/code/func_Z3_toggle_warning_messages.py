import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_toggle_warning_messages(a0, _elems=Elementaries(_lib.Z3_toggle_warning_messages)):
    _elems.f(a0)