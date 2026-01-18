import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def a_is_not_b(a, b):
    """
    This is `not (a is b)`
    """
    return a is not b