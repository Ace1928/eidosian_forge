import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
class TestOverloadPreferLiteral(TestCase):

    def test_overload(self):

        def prefer_lit(x):
            pass

        def non_lit(x):
            pass

        def ov(x):
            if isinstance(x, types.IntegerLiteral):
                if x.literal_value == 1:

                    def impl(x):
                        return 51966
                    return impl
                else:
                    raise errors.TypingError('literal value')
            else:

                def impl(x):
                    return x * 100
                return impl
        overload(prefer_lit, prefer_literal=True)(ov)
        overload(non_lit)(ov)

        @njit
        def check_prefer_lit(x):
            return (prefer_lit(1), prefer_lit(2), prefer_lit(x))
        a, b, c = check_prefer_lit(3)
        self.assertEqual(a, 51966)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)

        @njit
        def check_non_lit(x):
            return (non_lit(1), non_lit(2), non_lit(x))
        a, b, c = check_non_lit(3)
        self.assertEqual(a, 100)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)

    def test_overload_method(self):

        def ov(self, x):
            if isinstance(x, types.IntegerLiteral):
                if x.literal_value == 1:

                    def impl(self, x):
                        return 51966
                    return impl
                else:
                    raise errors.TypingError('literal value')
            else:

                def impl(self, x):
                    return x * 100
                return impl
        overload_method(MyDummyType, 'method_prefer_literal', prefer_literal=True)(ov)
        overload_method(MyDummyType, 'method_non_literal', prefer_literal=False)(ov)

        @njit
        def check_prefer_lit(dummy, x):
            return (dummy.method_prefer_literal(1), dummy.method_prefer_literal(2), dummy.method_prefer_literal(x))
        a, b, c = check_prefer_lit(MyDummy(), 3)
        self.assertEqual(a, 51966)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)

        @njit
        def check_non_lit(dummy, x):
            return (dummy.method_non_literal(1), dummy.method_non_literal(2), dummy.method_non_literal(x))
        a, b, c = check_non_lit(MyDummy(), 3)
        self.assertEqual(a, 100)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)