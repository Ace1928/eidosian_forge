from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
class PythonErrorModel(ErrorModel):
    """
    The Python error model.  Any invalid FP input raises an exception.
    """
    raise_on_fp_zero_division = True