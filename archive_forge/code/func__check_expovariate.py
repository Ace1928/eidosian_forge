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
def _check_expovariate(self, func, ptr):
    """
        Check a expovariate()-like function.  Note the second argument
        is inversed compared to np.random.exponential().
        """
    r = self._follow_numpy(ptr)
    for lambd in (0.2, 0.5, 1.5):
        for i in range(3):
            self.assertPreciseEqual(func(lambd), r.exponential(1 / lambd), prec='double')