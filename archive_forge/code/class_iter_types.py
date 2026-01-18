import operator as op
from functools import partial
import sys
import numpy as np
from .. import units as pq
from ..quantity import Quantity
from .common import TestCase
class iter_types:

    def __init__(self, dtype):
        self._index = -1
        self._dtype = dtype

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index > 2:
            raise StopIteration
        if self._index > 0 and self._dtype in (int, long, float, complex):
            raise StopIteration
        if self._index == 0:
            return rand(self._dtype)
        if self._index == 1:
            return rand(self._dtype, 5).tolist()
        if self._index == 2:
            return rand(self._dtype, 5)

    def next(self):
        return self.__next__()