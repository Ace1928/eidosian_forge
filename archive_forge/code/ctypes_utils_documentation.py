import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl

    Return a Numba type for the given ctypes function pointer.
    