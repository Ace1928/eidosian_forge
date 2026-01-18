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
class _ReflectContext(namedtuple('_ReflectContext', ('context', 'builder', 'pyapi', 'env_manager', 'is_error'))):
    """
    The facilities required by reflection implementations.
    """
    __slots__ = ()

    def set_error(self):
        self.builder.store(self.is_error, cgutils.true_bit)

    def box(self, typ, val):
        return self.pyapi.from_native_value(typ, val, self.env_manager)

    def reflect(self, typ, val):
        return self.pyapi.reflect_native_value(typ, val, self.env_manager)