import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def _createSampleaArray(self):
    self.refsample1d = np.recarray(3, dtype=recordwithcharseq)
    self.nbsample1d = np.zeros(3, dtype=recordwithcharseq)