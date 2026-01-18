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
def _check_uniform(self, func, ptr):
    """
        Check a uniform()-like function.
        """
    r = self._follow_cpython(ptr)
    self._check_dist(func, r.uniform, [(1.5, 1000000.0), (-2.5, 1000.0), (1.5, -2.5)])