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
def _check_array_dist_self(self, funcname, scalar_args):
    """
        Check function returning an array against its scalar implementation.
        Because we use the CPython gamma distribution rather than the NumPy one,
        distributions which use the gamma distribution vary in ways that are
        difficult to compare. Instead, we compile both the array and scalar
        versions and check that the array is filled with the same values as
        we would expect from the scalar version.
        """

    @numba.njit
    def reset():
        np.random.seed(1234)
    array_func = self._compile_array_dist(funcname, len(scalar_args) + 1)
    qualname = 'np.random.%s' % (funcname,)
    argstring = ', '.join('abcd'[:len(scalar_args)])
    scalar_func = jit_with_args(qualname, argstring)
    for size in (8, (2, 3)):
        args = scalar_args + (size,)
        reset()
        got = array_func(*args)
        reset()
        expected = np.empty(size, dtype=got.dtype)
        flat = expected.flat
        for idx in range(expected.size):
            flat[idx] = scalar_func(*scalar_args)
        self.assertPreciseEqual(expected, got, prec='double', ulps=5)
    reset()
    args = scalar_args + (None,)
    expected = scalar_func(*scalar_args)
    reset()
    got = array_func(*args)
    self.assertPreciseEqual(expected, got, prec='double', ulps=5)