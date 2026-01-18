import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
class ArrayCTypes(Type):
    """
    This is the type for `np.ndarray.ctypes`.
    """

    def __init__(self, arytype):
        self.dtype = arytype.dtype
        self.ndim = arytype.ndim
        name = 'ArrayCTypes(dtype={0}, ndim={1})'.format(self.dtype, self.ndim)
        super(ArrayCTypes, self).__init__(name)

    @property
    def key(self):
        return (self.dtype, self.ndim)

    def can_convert_to(self, typingctx, other):
        """
        Convert this type to the corresponding pointer type.
        This allows passing a array.ctypes object to a C function taking
        a raw pointer.

        Note that in pure Python, the array.ctypes object can only be
        passed to a ctypes function accepting a c_void_p, not a typed
        pointer.
        """
        from . import CPointer, voidptr
        if isinstance(other, CPointer) and other.dtype == self.dtype:
            return Conversion.safe
        elif other == voidptr:
            return Conversion.safe