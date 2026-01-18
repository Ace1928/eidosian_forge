from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support

    Add typing for a given CFFI type to the typemap
    