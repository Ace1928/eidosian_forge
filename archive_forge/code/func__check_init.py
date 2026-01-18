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
def _check_init(self, ptr):
    r = np.random.RandomState()
    for i in [0, 1, 125, 2 ** 32 - 5]:
        r.seed(np.uint32(i))
        st = r.get_state()
        ints = list(st[1])
        index = st[2]
        assert index == N
        _helperlib.rnd_seed(ptr, i)
        self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))