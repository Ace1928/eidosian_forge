import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
@njit
def MyStruct_get_vector(self):
    return self.vector