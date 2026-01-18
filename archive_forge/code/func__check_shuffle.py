import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def _check_shuffle(self, func, ptr, is_numpy):
    """
        Check a shuffle()-like function for arrays.
        """
    arrs = [np.arange(20), np.arange(32).reshape((8, 4))]
    if is_numpy:
        r = self._follow_numpy(ptr)
    else:
        r = self._follow_cpython(ptr)
    for a in arrs:
        for i in range(3):
            got = a.copy()
            expected = a.copy()
            func(got)
            if is_numpy or len(a.shape) == 1:
                r.shuffle(expected)
                self.assertPreciseEqual(got, expected)
    a = arrs[0]
    b = a.copy()
    func(memoryview(b))
    self.assertNotEqual(list(a), list(b))
    self.assertEqual(sorted(a), sorted(b))
    with self.assertTypingError():
        func(memoryview(b'xyz'))