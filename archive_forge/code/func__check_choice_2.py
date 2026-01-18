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
def _check_choice_2(self, a, pop):
    """
        Check choice(a, size) against pop.
        """
    cfunc = jit(nopython=True)(numpy_choice2)
    n = len(pop)
    sizes = [n - 10, (3, (n - 1) // 3), n * 10]
    for size in sizes:
        res = cfunc(a, size)
        expected_shape = size if isinstance(size, tuple) else (size,)
        self.assertEqual(res.shape, expected_shape)
        self._check_array_results(lambda: cfunc(a, size), pop)