from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
class TestListManagedElements(ManagedListTestCase):
    """Test list containing objects that need refct"""

    def _check_element_equal(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        con = [np.arange(3).astype(np.intp), np.arange(5).astype(np.intp)]
        expect = list(con)
        pyfunc(expect)
        got = list(con)
        cfunc(got)
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_reflect_passthru(self):

        def pyfunc(con):
            pass
        self._check_element_equal(pyfunc)

    def test_reflect_appended(self):

        def pyfunc(con):
            con.append(np.arange(10).astype(np.intp))
        self._check_element_equal(pyfunc)

    def test_reflect_setitem(self):

        def pyfunc(con):
            con[1] = np.arange(10)
        self._check_element_equal(pyfunc)

    def test_reflect_popped(self):

        def pyfunc(con):
            con.pop()
        self._check_element_equal(pyfunc)

    def test_reflect_insert(self):
        """make sure list.insert() doesn't crash for refcounted objects (see #7553)
        """

        def pyfunc(con):
            con.insert(1, np.arange(4).astype(np.intp))
        self._check_element_equal(pyfunc)

    def test_append(self):

        def pyfunc():
            con = []
            for i in range(300):
                con.append(np.arange(i).astype(np.intp))
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_append_noret(self):

        def pyfunc():
            con = []
            for i in range(300):
                con.append(np.arange(i))
            c = 0.0
            for arr in con:
                c += arr.sum() / (1 + arr.size)
            return c
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assertEqual(expect, got)

    def test_reassign_refct(self):

        def pyfunc():
            con = []
            for i in range(5):
                con.append(np.arange(2))
            con[0] = np.arange(4)
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_get_slice(self):

        def pyfunc():
            con = []
            for i in range(5):
                con.append(np.arange(2))
            return con[2:4]
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_set_slice(self):

        def pyfunc():
            con = []
            for i in range(5):
                con.append(np.arange(2))
            con[1:3] = con[2:4]
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_pop(self):

        def pyfunc():
            con = []
            for i in range(20):
                con.append(np.arange(i + 1))
            while len(con) > 2:
                con.pop()
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_pop_loc(self):

        def pyfunc():
            con = []
            for i in range(1000):
                con.append(np.arange(i + 1))
            while len(con) > 2:
                con.pop(1)
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_del_range(self):

        def pyfunc():
            con = []
            for i in range(20):
                con.append(np.arange(i + 1))
            del con[3:10]
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assert_list_element_precise_equal(expect=expect, got=got)

    def test_list_of_list(self):

        def pyfunc():
            con = []
            for i in range(10):
                con.append([0] * i)
            return con
        cfunc = jit(nopython=True)(pyfunc)
        expect = pyfunc()
        got = cfunc()
        self.assertEqual(expect, got)