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
@jitclass(spec)
class LinkedNode(object):

    def __init__(self, data, next):
        self.data = data
        self.next = next

    def get_next_data(self):
        return get_data(self.next)

    def append_to_tail(self, other):
        cur = self
        while cur.next is not None:
            cur = cur.next
        cur.next = other