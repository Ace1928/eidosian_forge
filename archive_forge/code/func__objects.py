import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def _objects(obj):
    objs = [obj]
    if isinstance(obj, tuple):
        for v in obj:
            objs += _objects(v)
    return objs