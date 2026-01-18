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
def _check_exponential(self, func1, func0, ptr):
    """
        Check a exponential()-like function.
        """
    r = self._follow_numpy(ptr)
    if func1 is not None:
        self._check_dist(func1, r.exponential, [(0.5,), (1.0,), (1.5,)])
    if func0 is not None:
        self._check_dist(func0, r.exponential, [()])