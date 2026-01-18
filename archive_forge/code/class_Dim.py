from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
class Dim(object):
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """
    __slots__ = ('start', 'stop', 'size', 'stride', 'single')

    def __init__(self, start, stop, size, stride, single):
        self.start = start
        self.stop = stop
        self.size = size
        self.stride = stride
        self.single = single
        assert not single or size == 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.size)
            stride = step * self.stride
            start = self.start + start * abs(self.stride)
            stop = self.start + stop * abs(self.stride)
            if stride == 0:
                size = 1
            else:
                size = _compute_size(start, stop, stride)
            ret = Dim(start=start, stop=stop, size=size, stride=stride, single=False)
            return ret
        else:
            sliced = self[item:item + 1] if item != -1 else self[-1:]
            if sliced.size != 1:
                raise IndexError
            return Dim(start=sliced.start, stop=sliced.stop, size=sliced.size, stride=sliced.stride, single=True)

    def get_offset(self, idx):
        return self.start + idx * self.stride

    def __repr__(self):
        strfmt = 'Dim(start=%s, stop=%s, size=%s, stride=%s)'
        return strfmt % (self.start, self.stop, self.size, self.stride)

    def normalize(self, base):
        return Dim(start=self.start - base, stop=self.stop - base, size=self.size, stride=self.stride, single=self.single)

    def copy(self, start=None, stop=None, size=None, stride=None, single=None):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if size is None:
            size = self.size
        if stride is None:
            stride = self.stride
        if single is None:
            single = self.single
        return Dim(start, stop, size, stride, single)

    def is_contiguous(self, itemsize):
        return self.stride == itemsize