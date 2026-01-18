from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def _subshape(self, index):
    """ The DataShape of an indexed subarray

        >>> from datashader.datashape import dshape

        >>> ds = dshape('var * {name: string, amount: int32}')
        >>> print(ds.subshape[0])
        {name: string, amount: int32}

        >>> print(ds.subshape[0:3])
        3 * {name: string, amount: int32}

        >>> print(ds.subshape[0:7:2, 'amount'])
        4 * int32

        >>> print(ds.subshape[[1, 10, 15]])
        3 * {name: string, amount: int32}

        >>> ds = dshape('{x: int, y: int}')
        >>> print(ds.subshape['x'])
        int32

        >>> ds = dshape('10 * var * 10 * int32')
        >>> print(ds.subshape[0:5, 0:3, 5])
        5 * 3 * int32

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, [0, 2]])
        var * {name: string, id: int32}

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, ['name', 'id']])
        var * {name: string, id: int32}

        >>> print(ds.subshape[0, 1:])
        {amount: int32, id: int32}
        """
    from .predicates import isdimension
    if isinstance(index, int) and isdimension(self[0]):
        return self.subarray(1)
    if isinstance(self[0], Record) and isinstance(index, str):
        return self[0][index]
    if isinstance(self[0], Record) and isinstance(index, int):
        return self[0].parameters[0][index][1]
    if isinstance(self[0], Record) and isinstance(index, list):
        rec = self[0]
        index = [self[0].names.index(i) if isinstance(i, str) else i for i in index]
        return DataShape(Record([rec.parameters[0][i] for i in index]))
    if isinstance(self[0], Record) and isinstance(index, slice):
        rec = self[0]
        return DataShape(Record(rec.parameters[0][index]))
    if isinstance(index, list) and isdimension(self[0]):
        return len(index) * self.subarray(1)
    if isinstance(index, slice):
        if isinstance(self[0], Fixed):
            n = int(self[0])
            start = index.start or 0
            stop = index.stop or n
            if start < 0:
                start = n + start
            if stop < 0:
                stop = n + stop
            count = stop - start
        else:
            start = index.start or 0
            stop = index.stop
            if not stop:
                count = -start if start < 0 else var
            if stop is not None and start is not None and (stop >= 0) and (start >= 0):
                count = stop - start
            else:
                count = var
        if count != var and index.step is not None:
            count = int(ceil(count / index.step))
        return count * self.subarray(1)
    if isinstance(index, tuple):
        if not index:
            return self
        elif index[0] is None:
            return 1 * self._subshape(index[1:])
        elif len(index) == 1:
            return self._subshape(index[0])
        else:
            ds = self.subarray(1)._subshape(index[1:])
            return (self[0] * ds)._subshape(index[0])
    raise TypeError('invalid index value %s of type %r' % (index, type(index).__name__))