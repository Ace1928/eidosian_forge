import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestUnicodeSets(TestSets):
    """
    Test sets with unicode keys. For the purpose of testing refcounted sets.
    """

    def _range(self, stop):
        return ['A{}'.format(i) for i in range(int(stop))]