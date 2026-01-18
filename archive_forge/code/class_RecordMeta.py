from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
class RecordMeta(Type):

    @staticmethod
    def _unpack_slice(s, idx):
        if not isinstance(s, slice):
            raise TypeError('invalid field specification at position %d.\nfields must be formatted like: {name}:{type}' % idx)
        name, type_ = packed = (s.start, s.stop)
        if name is None:
            raise TypeError('missing field name at position %d' % idx)
        if not isinstance(name, str):
            raise TypeError("field name at position %d ('%s') was not a string" % (idx, name))
        if type_ is None and s.step is None:
            raise TypeError("missing type for field '%s' at position %d" % (name, idx))
        if s.step is not None:
            raise TypeError("unexpected slice step for field '%s' at position %d.\nhint: you might have a second ':'" % (name, idx))
        return packed

    def __getitem__(self, types):
        if not isinstance(types, tuple):
            types = (types,)
        return self(list(map(self._unpack_slice, types, range(len(types)))))