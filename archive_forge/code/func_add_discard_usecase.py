import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def add_discard_usecase(a, u, v):
    s = set(a)
    for i in range(1000):
        s.add(u)
        s.discard(v)
    return list(s)