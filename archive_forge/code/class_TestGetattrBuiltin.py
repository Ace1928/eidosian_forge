import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
class TestGetattrBuiltin(MemoryLeakMixin, TestCase):

    def test_getattr_func_retty(self):

        @njit
        def foo(x):
            attr = getattr(x, '__hash__')
            return attr()
        for x in (1, 2.34, (5, 6, 7)):
            self.assertPreciseEqual(foo(x), foo.py_func(x))

    def test_getattr_value_retty(self):

        @njit
        def foo(x):
            return getattr(x, 'ndim')
        for x in range(3):
            tmp = np.empty((1,) * x)
            self.assertPreciseEqual(foo(tmp), foo.py_func(tmp))

    def test_getattr_module_obj(self):

        @njit
        def foo():
            return getattr(np, 'pi')
        self.assertPreciseEqual(foo(), foo.py_func())

    def test_getattr_module_obj_not_implemented(self):

        @njit
        def foo():
            return getattr(np, 'cos')(1)
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        msg = 'Returning function objects is not implemented'
        self.assertIn(msg, str(raises.exception))

    def test_getattr_raises_attribute_error(self):
        invalid_attr = '__not_a_valid_attr__'

        @njit
        def foo(x):
            return getattr(x, invalid_attr)
        with self.assertRaises(AttributeError) as raises:
            foo(1.23)
        self.assertIn(f"'float64' has no attribute '{invalid_attr}'", str(raises.exception))

    def test_getattr_with_default(self):

        @njit
        def foo(x, default):
            return getattr(x, '__not_a_valid_attr__', default)
        for x, y in zip((1, 2.34, (5, 6, 7)), (None, 20, 'some_string')):
            self.assertPreciseEqual(foo(x, y), foo.py_func(x, y))

    def test_getattr_non_literal_str(self):

        @njit
        def foo(x, nonliteral_str):
            return getattr(x, nonliteral_str)
        with self.assertRaises(errors.TypingError) as raises:
            foo(1, '__hash__')
        msg = "argument 'name' must be a literal string"
        self.assertIn(msg, str(raises.exception))

    def test_getattr_no_optional_type_generated(self):

        @njit
        def default_hash():
            return 12345

        @njit
        def foo():
            hash_func = getattr(np.ones(1), '__not_a_valid_attr__', default_hash)
            return hash_func()
        self.assertPreciseEqual(foo(), foo.py_func())