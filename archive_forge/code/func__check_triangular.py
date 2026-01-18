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
def _check_triangular(self, func2, func3, ptr):
    """
        Check a triangular()-like function.
        """
    r = self._follow_cpython(ptr)
    if func2 is not None:
        self._check_dist(func2, r.triangular, [(1.5, 3.5), (-2.5, 1.5), (1.5, 1.5)])
    self._check_dist(func3, r.triangular, [(1.5, 3.5, 2.2)])