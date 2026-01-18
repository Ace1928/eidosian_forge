from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
class _UnboxContext(namedtuple('_UnboxContext', ('context', 'builder', 'pyapi'))):
    """
    The facilities required by unboxing implementations.
    """
    __slots__ = ()

    def unbox(self, typ, obj):
        return self.pyapi.to_native_value(typ, obj)