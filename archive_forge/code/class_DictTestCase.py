import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
class DictTestCase(TestCase):

    def check(self, pyfunc):
        cfunc = jit(forceobj=True)(pyfunc)
        self.assertPreciseEqual(pyfunc(), cfunc())

    def test_build_map(self):
        self.check(build_map)

    def test_build_map_from_local_vars(self):
        self.check(build_map_from_local_vars)