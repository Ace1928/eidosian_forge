import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
class TestBoundsEnvironmentVariable(TestCase):

    def setUp(self):

        @njit
        def default(x):
            return x[1]

        @njit(boundscheck=False)
        def off(x):
            return x[1]

        @njit(boundscheck=True)
        def on(x):
            return x[1]
        self.default = default
        self.off = off
        self.on = on

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
    def test_boundscheck_unset(self):
        self.assertIsNone(config.BOUNDSCHECK)
        a = np.array([1])
        self.default(a)
        self.off(a)
        with self.assertRaises(IndexError):
            self.on(a)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '1'})
    def test_boundscheck_enabled(self):
        self.assertTrue(config.BOUNDSCHECK)
        a = np.array([1])
        with self.assertRaises(IndexError):
            self.default(a)
            self.off(a)
            self.on(a)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '0'})
    def test_boundscheck_disabled(self):
        self.assertFalse(config.BOUNDSCHECK)
        a = np.array([1])
        self.default(a)
        self.off(a)
        self.on(a)