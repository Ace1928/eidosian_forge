import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def _make_Float2AndArray(self):
    spec = OrderedDict()
    spec['x'] = float32
    spec['y'] = float32
    spec['arr'] = float32[:]

    @jitclass(spec)
    class Float2AndArray(object):

        def __init__(self, x, y, arr):
            self.x = x
            self.y = y
            self.arr = arr

        def add(self, val):
            self.x += val
            self.y += val
            return val
    return Float2AndArray