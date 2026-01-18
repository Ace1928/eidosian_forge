import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
class _ndptr(_ndptr_base):

    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError('argument must be an ndarray')
        if cls._dtype_ is not None and obj.dtype != cls._dtype_:
            raise TypeError('array must have data type %s' % cls._dtype_)
        if cls._ndim_ is not None and obj.ndim != cls._ndim_:
            raise TypeError('array must have %d dimension(s)' % cls._ndim_)
        if cls._shape_ is not None and obj.shape != cls._shape_:
            raise TypeError('array must have shape %s' % str(cls._shape_))
        if cls._flags_ is not None and obj.flags.num & cls._flags_ != cls._flags_:
            raise TypeError('array must have flags %s' % _flags_fromnum(cls._flags_))
        return obj.ctypes