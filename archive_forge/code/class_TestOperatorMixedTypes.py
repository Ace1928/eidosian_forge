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
class TestOperatorMixedTypes(TestCase):

    def test_eq_ne(self):
        for opstr in ('eq', 'ne'):
            op = getattr(operator, opstr)

            @njit
            def func(a, b):
                return op(a, b)
            things = (1, 0, True, False, 1.0, 2.0, 1.1, 1j, None, '', '1')
            for x, y in itertools.product(things, things):
                self.assertPreciseEqual(func.py_func(x, y), func(x, y))

    def test_cmp(self):
        for opstr in ('gt', 'lt', 'ge', 'le', 'eq', 'ne'):
            op = getattr(operator, opstr)

            @njit
            def func(a, b):
                return op(a, b)
            things = (1, 0, True, False, 1.0, 0.0, 1.1)
            for x, y in itertools.product(things, things):
                expected = func.py_func(x, y)
                got = func(x, y)
                message = '%s %s %s does not match between Python and Numba' % (x, opstr, y)
                self.assertEqual(expected, got, message)