import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
class TestTypedDictInitialValues(MemoryLeakMixin, TestCase):
    """Tests that typed dictionaries carry their initial value if present"""

    def test_homogeneous_and_literal(self):

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, {'a': 1, 'b': 2, 'c': 3})
            self.assertEqual(hasattr(d, 'literal_value'), False)
            return lambda d: d

        @njit
        def foo():
            x = {'a': 1, 'b': 2, 'c': 3}
            bar(x)
        foo()

    def test_heterogeneous_but_castable_to_homogeneous(self):

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, None)
            self.assertEqual(hasattr(d, 'literal_value'), False)
            return lambda d: d

        @njit
        def foo():
            x = {'a': 1j, 'b': 2, 'c': 3}
            bar(x)
        foo()

    def test_heterogeneous_but_not_castable_to_homogeneous(self):

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            a = {'a': 1, 'b': 2j, 'c': 3}

            def specific_ty(z):
                return types.literal(z) if types.maybe_literal(z) else typeof(z)
            expected = {types.literal(x): specific_ty(y) for x, y in a.items()}
            self.assertTrue(isinstance(d, types.LiteralStrKeyDict))
            self.assertEqual(d.literal_value, expected)
            self.assertEqual(hasattr(d, 'initial_value'), False)
            return lambda d: d

        @njit
        def foo():
            x = {'a': 1, 'b': 2j, 'c': 3}
            bar(x)
        foo()

    def test_mutation_not_carried(self):

        def bar(d):
            ...

        @overload(bar)
        def ol_bar(d):
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, {'a': 1, 'b': 2, 'c': 3})
            return lambda d: d

        @njit
        def foo():
            x = {'a': 1, 'b': 2, 'c': 3}
            x['d'] = 4
            bar(x)
        foo()

    def test_mutation_not_carried_single_function(self):

        @njit
        def nop(*args):
            pass
        for fn, iv in ((nop, None), (literally, {'a': 1, 'b': 2, 'c': 3})):

            @njit
            def baz(x):
                pass

            def bar(z):
                pass

            @overload(bar)
            def ol_bar(z):

                def impl(z):
                    fn(z)
                    baz(z)
                return impl

            @njit
            def foo():
                x = {'a': 1, 'b': 2, 'c': 3}
                bar(x)
                x['d'] = 4
                return x
            foo()
            larg = baz.signatures[0][0]
            self.assertEqual(larg.initial_value, iv)

    def test_unify_across_function_call(self):

        @njit
        def bar(x):
            o = {1: 2}
            if x:
                o = {2: 3}
            return o

        @njit
        def foo(x):
            if x:
                d = {3: 4}
            else:
                d = bar(x)
            return d
        e1 = Dict()
        e1[3] = 4
        e2 = Dict()
        e2[1] = 2
        self.assertEqual(foo(True), e1)
        self.assertEqual(foo(False), e2)