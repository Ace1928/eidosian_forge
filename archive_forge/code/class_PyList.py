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
class PyList:

    def __init__(self):
        self.x = [0]

    def append(self, y):
        self.x.append(y)

    def clear(self):
        self.x.clear()

    def __abs__(self):
        return len(self.x) * 7

    def __bool__(self):
        return len(self.x) % 3 != 0

    def __complex__(self):
        c = complex(2)
        if self.x:
            c += self.x[0]
        return c

    def __contains__(self, y):
        return y in self.x

    def __float__(self):
        f = 3.1415
        if self.x:
            f += self.x[0]
        return f

    def __int__(self):
        i = 5
        if self.x:
            i += self.x[0]
        return i

    def __len__(self):
        return len(self.x) + 1

    def __str__(self):
        if len(self.x) == 0:
            return 'PyList empty'
        else:
            return 'PyList non-empty'