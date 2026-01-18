import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
class SeriesType(types.ArrayCompatible):
    """
    The type class for Series objects.
    """
    array_priority = 1000

    def __init__(self, dtype, index):
        assert isinstance(index, IndexType)
        self.dtype = dtype
        self.index = index
        self.values = types.Array(self.dtype, 1, 'C')
        name = 'series(%s, %s)' % (dtype, index)
        super(SeriesType, self).__init__(name)

    @property
    def key(self):
        return (self.dtype, self.index)

    @property
    def as_array(self):
        return self.values

    def copy(self, dtype=None, ndim=1, layout='C'):
        assert ndim == 1
        assert layout == 'C'
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype, self.index)