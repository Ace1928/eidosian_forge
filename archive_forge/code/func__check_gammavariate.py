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
def _check_gammavariate(self, func2, func1, ptr):
    """
        Check a gammavariate()-like function.
        """
    r = self._follow_cpython(ptr)
    if func2 is not None:
        self._check_dist(func2, r.gammavariate, [(0.5, 2.5), (1.0, 1.5), (1.5, 3.5)])
    if func1 is not None:
        self.assertPreciseEqual(func1(1.5), r.gammavariate(1.5, 1.0))
    if func2 is not None:
        self.assertRaises(ValueError, func2, 0.0, 1.0)
        self.assertRaises(ValueError, func2, 1.0, 0.0)
        self.assertRaises(ValueError, func2, -0.5, 1.0)
        self.assertRaises(ValueError, func2, 1.0, -0.5)
    if func1 is not None:
        self.assertRaises(ValueError, func1, 0.0)
        self.assertRaises(ValueError, func1, -0.5)