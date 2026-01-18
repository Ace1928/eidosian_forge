import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
class TestGeneratorWithNRT(MemoryLeakMixin, TestCase):

    def test_issue_1254(self):
        """
        Missing environment for returning array
        """

        @jit(nopython=True)
        def random_directions(n):
            for i in range(n):
                vec = np.empty(3)
                vec[:] = 12
                yield vec
        outputs = list(random_directions(5))
        self.assertEqual(len(outputs), 5)
        expect = np.empty(3)
        expect[:] = 12
        for got in outputs:
            np.testing.assert_equal(expect, got)

    def test_issue_1265(self):
        """
        Double-free for locally allocated, non escaping NRT objects
        """

        def py_gen(rmin, rmax, nr):
            a = np.linspace(rmin, rmax, nr)
            yield a[0]
            yield a[1]
        c_gen = jit(nopython=True)(py_gen)
        py_res = list(py_gen(-2, 2, 100))
        c_res = list(c_gen(-2, 2, 100))
        self.assertEqual(py_res, c_res)

        def py_driver(args):
            rmin, rmax, nr = args
            points = np.empty(nr, dtype=np.complex128)
            for i, c in enumerate(py_gen(rmin, rmax, nr)):
                points[i] = c
            return points

        @jit(nopython=True)
        def c_driver(args):
            rmin, rmax, nr = args
            points = np.empty(nr, dtype=np.complex128)
            for i, c in enumerate(c_gen(rmin, rmax, nr)):
                points[i] = c
            return points
        n = 2
        patches = (-2, -1, n)
        py_res = py_driver(patches)
        c_res = c_driver(patches)
        np.testing.assert_equal(py_res, c_res)

    def test_issue_1808(self):
        """
        Incorrect return data model
        """
        magic = 3735928559

        @njit
        def generator():
            yield magic

        @njit
        def get_generator():
            return generator()

        @njit
        def main():
            out = 0
            for x in get_generator():
                out += x
            return out
        self.assertEqual(main(), magic)