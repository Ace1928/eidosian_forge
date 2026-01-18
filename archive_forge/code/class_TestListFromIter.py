import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
class TestListFromIter(MemoryLeakMixin, TestCase):

    def test_simple_iterable_types(self):
        """Test all simple iterables that a List can be constructed from."""

        def generate_function(line):
            context = {}
            code = dedent('\n                from numba.typed import List\n                def bar():\n                    {}\n                    return l\n                ').format(line)
            exec(code, context)
            return njit(context['bar'])
        for line in ('l = List([0, 1, 2])', 'l = List(range(3))', 'l = List(List([0, 1, 2]))', 'l = List((0, 1, 2))', 'l = List(set([0, 1, 2]))'):
            foo = generate_function(line)
            cf_received, py_received = (foo(), foo.py_func())
            for result in (cf_received, py_received):
                for i in range(3):
                    self.assertEqual(i, result[i])

    def test_unicode(self):
        """Test that a List can be created from a unicode string."""

        @njit
        def foo():
            l = List('abc')
            return l
        expected = List()
        for i in ('a', 'b', 'c'):
            expected.append(i)
        self.assertEqual(foo.py_func(), expected)
        self.assertEqual(foo(), expected)

    def test_dict_iters(self):
        """Test that a List can be created from Dict iterators."""

        def generate_function(line):
            context = {}
            code = dedent('\n                from numba.typed import List, Dict\n                def bar():\n                    d = Dict()\n                    d[0], d[1], d[2] = "a", "b", "c"\n                    {}\n                    return l\n                ').format(line)
            exec(code, context)
            return njit(context['bar'])

        def generate_expected(values):
            expected = List()
            for i in values:
                expected.append(i)
            return expected
        for line, values in (('l = List(d)', (0, 1, 2)), ('l = List(d.keys())', (0, 1, 2)), ('l = List(d.values())', ('a', 'b', 'c')), ('l = List(d.items())', ((0, 'a'), (1, 'b'), (2, 'c')))):
            foo, expected = (generate_function(line), generate_expected(values))
            for func in (foo, foo.py_func):
                self.assertEqual(func(), expected)

    def test_ndarray_scalar(self):

        @njit
        def foo():
            return List(np.ones(3))
        expected = List()
        for i in range(3):
            expected.append(1)
        self.assertEqual(expected, foo())
        self.assertEqual(expected, foo.py_func())

    def test_ndarray_oned(self):

        @njit
        def foo():
            return List(np.array(1))
        expected = List()
        expected.append(1)
        self.assertEqual(expected, foo())
        self.assertEqual(expected, foo.py_func())

    def test_ndarray_twod(self):

        @njit
        def foo(x):
            return List(x)
        carr = np.array([[1, 2], [3, 4]])
        farr = np.asfortranarray(carr)
        aarr = np.arange(8).reshape((2, 4))[:, ::2]
        for layout, arr in zip('CFA', (carr, farr, aarr)):
            self.assertEqual(typeof(arr).layout, layout)
            expected = List()
            expected.append(arr[0, :])
            expected.append(arr[1, :])
            received = foo(arr)
            np.testing.assert_equal(expected[0], received[0])
            np.testing.assert_equal(expected[1], received[1])
            pyreceived = foo.py_func(arr)
            np.testing.assert_equal(expected[0], pyreceived[0])
            np.testing.assert_equal(expected[1], pyreceived[1])

    def test_exception_on_plain_int(self):

        @njit
        def foo():
            l = List(23)
            return l
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('List() argument must be iterable', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            List(23)
        self.assertIn('List() argument must be iterable', str(raises.exception))

    def test_exception_on_inhomogeneous_tuple(self):

        @njit
        def foo():
            l = List((1, 1.0))
            return l
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('List() argument must be iterable', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            List((1, 1.0))

    def test_exception_on_too_many_args(self):

        @njit
        def foo():
            l = List((0, 1, 2), (3, 4, 5))
            return l
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('List() expected at most 1 argument, got 2', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            List((0, 1, 2), (3, 4, 5))
        self.assertIn('List() expected at most 1 argument, got 2', str(raises.exception))

        @njit
        def foo():
            l = List((0, 1, 2), (3, 4, 5), (6, 7, 8))
            return l
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('List() expected at most 1 argument, got 3', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            List((0, 1, 2), (3, 4, 5), (6, 7, 8))
        self.assertIn('List() expected at most 1 argument, got 3', str(raises.exception))

    def test_exception_on_kwargs(self):

        @njit
        def foo():
            l = List(iterable=(0, 1, 2))
            return l
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('List() takes no keyword arguments', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            List(iterable=(0, 1, 2))
        self.assertIn('List() takes no keyword arguments', str(raises.exception))