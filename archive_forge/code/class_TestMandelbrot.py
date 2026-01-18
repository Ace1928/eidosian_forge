import unittest
from numba import njit
from numba.core import types
class TestMandelbrot(unittest.TestCase):

    def test_mandelbrot(self):
        pyfunc = is_in_mandelbrot
        cfunc = njit((types.complex64,))(pyfunc)
        points = [0 + 0j, 1 + 0j, 0 + 1j, 1 + 1j, 0.1 + 0.1j]
        for p in points:
            self.assertEqual(cfunc(p), pyfunc(p))