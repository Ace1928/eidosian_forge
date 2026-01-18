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
def _check_array_dist_gamma(self, funcname, scalar_args, extra_pyfunc_args):
    """
        Check returning an array according to a given gamma distribution,
        where we use CPython's implementation rather than NumPy's.
        """
    cfunc = self._compile_array_dist(funcname, len(scalar_args) + 1)
    r = self._follow_cpython(get_np_state_ptr())
    pyfunc = getattr(r, 'gammavariate')
    pyfunc_args = scalar_args + extra_pyfunc_args
    pyrandom = lambda *_args: pyfunc(*pyfunc_args)
    args = scalar_args + (None,)
    expected = pyrandom()
    got = cfunc(*args)
    self.assertPreciseEqual(expected, got, prec='double', ulps=5)
    for size in (8, (2, 3)):
        args = scalar_args + (size,)
        expected = np.empty(size)
        expected_flat = expected.flat
        for idx in range(expected.size):
            expected_flat[idx] = pyrandom()
        got = cfunc(*args)
        self.assertPreciseEqual(expected, got, prec='double', ulps=5)