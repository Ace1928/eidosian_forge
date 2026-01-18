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
class OperatorsDefined:

    def __init__(self, x):
        self.x = x

    def __radd__(self, other):
        return other.x + self.x

    def __rsub__(self, other):
        return other.x - self.x

    def __rmul__(self, other):
        return other.x * self.x

    def __rtruediv__(self, other):
        return other.x / self.x

    def __rfloordiv__(self, other):
        return other.x // self.x

    def __rmod__(self, other):
        return other.x % self.x

    def __rpow__(self, other):
        return other.x ** self.x

    def __rlshift__(self, other):
        return other.x << self.x

    def __rrshift__(self, other):
        return other.x >> self.x

    def __rand__(self, other):
        return other.x & self.x

    def __rxor__(self, other):
        return other.x ^ self.x

    def __ror__(self, other):
        return other.x | self.x