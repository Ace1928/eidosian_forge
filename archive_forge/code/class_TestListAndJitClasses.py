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
class TestListAndJitClasses(ManagedListTestCase):

    def make_jitclass_element(self):
        spec = [('many', types.float64[:]), ('scalar', types.float64)]
        JCItem = jitclass(spec)(Item)
        return JCItem

    def make_jitclass_container(self):
        spec = {'data': types.List(dtype=types.List(types.float64[::1]))}
        JCContainer = jitclass(spec)(Container)
        return JCContainer

    def assert_list_element_with_tester(self, tester, expect, got):
        for x, y in zip(expect, got):
            tester(x, y)

    def test_jitclass_instance_elements(self):
        JCItem = self.make_jitclass_element()

        def pyfunc(xs):
            xs[1], xs[0] = (xs[0], xs[1])
            return xs

        def eq(x, y):
            self.assertPreciseEqual(x.many, y.many)
            self.assertPreciseEqual(x.scalar, y.scalar)
        cfunc = jit(nopython=True)(pyfunc)
        arg = [JCItem(many=np.random.random(n + 1), scalar=n * 1.2) for n in range(5)]
        expect_arg = list(arg)
        got_arg = list(arg)
        expect_res = pyfunc(expect_arg)
        got_res = cfunc(got_arg)
        self.assert_list_element_with_tester(eq, expect_arg, got_arg)
        self.assert_list_element_with_tester(eq, expect_res, got_res)

    def test_jitclass_containing_list(self):
        JCContainer = self.make_jitclass_container()
        expect = Container(n=4)
        got = JCContainer(n=4)
        self.assert_list_element_precise_equal(got.data, expect.data)
        expect.more(3)
        got.more(3)
        self.assert_list_element_precise_equal(got.data, expect.data)