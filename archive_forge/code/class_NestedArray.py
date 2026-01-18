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
class NestedArray(Array):
    """
    A NestedArray is an array nested within a structured type (which are "void"
    type in NumPy parlance). Unlike an Array, the shape, and not just the number
    of dimensions is part of the type of a NestedArray.
    """

    def __init__(self, dtype, shape):
        if isinstance(dtype, NestedArray):
            tmp = Array(dtype.dtype, dtype.ndim, 'C')
            shape += dtype.shape
            dtype = tmp.dtype
        assert dtype.bitwidth % 8 == 0, 'Dtype bitwidth must be a multiple of bytes'
        self._shape = shape
        name = 'nestedarray(%s, %s)' % (dtype, shape)
        ndim = len(shape)
        super(NestedArray, self).__init__(dtype, ndim, 'C', name=name)

    @property
    def shape(self):
        return self._shape

    @property
    def nitems(self):
        l = 1
        for s in self.shape:
            l = l * s
        return l

    @property
    def size(self):
        return self.dtype.bitwidth // 8

    @property
    def strides(self):
        stride = self.size
        strides = []
        for i in reversed(self._shape):
            strides.append(stride)
            stride *= i
        return tuple(reversed(strides))

    @property
    def key(self):
        return (self.dtype, self.shape)

    def __repr__(self):
        return f'NestedArray({repr(self.dtype)}, {self.shape})'