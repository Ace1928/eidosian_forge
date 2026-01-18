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
class JitIntUpdateWrapper(PyIntWrapper):

    def __init__(self, value):
        self.x = value

    def __ilshift__(self, other):
        return JitIntUpdateWrapper(self.x << other.x)

    def __irshift__(self, other):
        return JitIntUpdateWrapper(self.x >> other.x)

    def __iand__(self, other):
        return JitIntUpdateWrapper(self.x & other.x)

    def __ior__(self, other):
        return JitIntUpdateWrapper(self.x | other.x)

    def __ixor__(self, other):
        return JitIntUpdateWrapper(self.x ^ other.x)