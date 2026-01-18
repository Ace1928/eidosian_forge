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
def _copy_py_state(r, ptr):
    """
    Copy state of Python random *r* to Numba state *ptr*.
    """
    mt = r.getstate()[1]
    ints, index = (mt[:-1], mt[-1])
    _helperlib.rnd_set_state(ptr, (index, list(ints)))
    return (ints, index)