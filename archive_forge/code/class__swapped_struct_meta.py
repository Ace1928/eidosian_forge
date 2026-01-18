import sys
from ctypes import *
class _swapped_struct_meta(_swapped_meta, type(Structure)):
    pass