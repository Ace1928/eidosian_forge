from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
class FakeCUDAAtomic(object):

    def add(self, array, index, val):
        with addlock:
            old = array[index]
            array[index] += val
        return old

    def sub(self, array, index, val):
        with sublock:
            old = array[index]
            array[index] -= val
        return old

    def and_(self, array, index, val):
        with andlock:
            old = array[index]
            array[index] &= val
        return old

    def or_(self, array, index, val):
        with orlock:
            old = array[index]
            array[index] |= val
        return old

    def xor(self, array, index, val):
        with xorlock:
            old = array[index]
            array[index] ^= val
        return old

    def inc(self, array, index, val):
        with inclock:
            old = array[index]
            if old >= val:
                array[index] = 0
            else:
                array[index] += 1
        return old

    def dec(self, array, index, val):
        with declock:
            old = array[index]
            if old == 0 or old > val:
                array[index] = val
            else:
                array[index] -= 1
        return old

    def exch(self, array, index, val):
        with exchlock:
            old = array[index]
            array[index] = val
        return old

    def max(self, array, index, val):
        with maxlock:
            old = array[index]
            array[index] = max(old, val)
        return old

    def min(self, array, index, val):
        with minlock:
            old = array[index]
            array[index] = min(old, val)
        return old

    def nanmax(self, array, index, val):
        with maxlock:
            old = array[index]
            array[index] = np.nanmax([array[index], val])
        return old

    def nanmin(self, array, index, val):
        with minlock:
            old = array[index]
            array[index] = np.nanmin([array[index], val])
        return old

    def compare_and_swap(self, array, old, val):
        with compare_and_swaplock:
            index = (0,) * array.ndim
            loaded = array[index]
            if loaded == old:
                array[index] = val
            return loaded

    def cas(self, array, index, old, val):
        with caslock:
            loaded = array[index]
            if loaded == old:
                array[index] = val
            return loaded