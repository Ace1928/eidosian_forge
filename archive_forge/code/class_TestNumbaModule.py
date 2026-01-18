import warnings
import numba
from numba import jit, njit
from numba.tests.support import TestCase, always_test
import unittest
class TestNumbaModule(TestCase):
    """
    Test the APIs exposed by the top-level `numba` module.
    """

    def check_member(self, name):
        self.assertTrue(hasattr(numba, name), name)
        self.assertIn(name, numba.__all__)

    @always_test
    def test_numba_module(self):
        self.check_member('jit')
        self.check_member('vectorize')
        self.check_member('guvectorize')
        self.check_member('njit')
        self.check_member('NumbaError')
        self.check_member('TypingError')
        self.check_member('int32')
        numba.__version__