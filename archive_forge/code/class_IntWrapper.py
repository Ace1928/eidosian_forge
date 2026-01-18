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
@jitclass([('x', types.intp)])
class IntWrapper:

    def __init__(self, value):
        self.x = value

    def __eq__(self, other):
        return self.x == other.x

    def __hash__(self):
        return self.x

    def __lshift__(self, other):
        return IntWrapper(self.x << other.x)

    def __rshift__(self, other):
        return IntWrapper(self.x >> other.x)

    def __and__(self, other):
        return IntWrapper(self.x & other.x)

    def __or__(self, other):
        return IntWrapper(self.x | other.x)

    def __xor__(self, other):
        return IntWrapper(self.x ^ other.x)