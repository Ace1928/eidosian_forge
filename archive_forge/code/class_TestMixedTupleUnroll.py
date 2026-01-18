from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
class TestMixedTupleUnroll(MemoryLeakMixin, TestCase):

    def test_01(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for i in range(len(literal_unroll(a))):
                acc += a[i]
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc
        f = 9
        k = f
        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_02(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for a in literal_unroll(x):
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc
        f = 9
        k = f
        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_03(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            y = ('foo', z, 2 * z)
            acc = 0
            for a in literal_unroll(x):
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    for t in literal_unroll(y):
                        acc += t is False
                    break
            return acc
        f = 9
        k = f
        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_04(self):

        @njit
        def foo(tup):
            acc = 0
            for a in literal_unroll(tup):
                acc += a.sum()
            return acc
        n = 10
        tup = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)))
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_05(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            for a in literal_unroll(tup1):
                if a == 'a':
                    acc += tup2[0].sum()
                elif a == 'b':
                    acc += tup2[1].sum()
                elif a == 'c':
                    acc += tup2[2].sum()
                elif a == 12:
                    acc += tup2[3].sum()
                elif a == 3j:
                    acc += tup2[4].sum()
                else:
                    raise RuntimeError('Unreachable')
            return acc
        n = 10
        tup1 = ('a', 'b', 'c', 12, 3j)
        tup2 = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)), np.ones((n, n, n, n)), np.ones((n, n, n, n, n)))
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

    @unittest.skip('needs more clever branch prune')
    def test_06(self):

        @njit
        def foo(tup):
            acc = 0
            str_buf = typed.List.empty_list(types.unicode_type)
            for a in literal_unroll(tup):
                if a == 'a':
                    str_buf.append(a)
                else:
                    acc += a
            return acc
        tup = ('a', 12)
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_07(self):

        @njit
        def foo(tup):
            acc = 0
            for a in literal_unroll(tup):
                acc += len(a)
            return acc
        n = 10
        tup = (np.ones((n,)), np.ones((n, n)), 'ABCDEFGHJI', (1, 2, 3), (1, 'foo', 2, 'bar'), {3, 4, 5, 6, 7})
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_08(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            for a in literal_unroll(tup1):
                if a == 'a':
                    acc += tup2[0]()
                elif a == 'b':
                    acc += tup2[1]()
                elif a == 'c':
                    acc += tup2[2]()
            return acc

        def gen(x):

            def impl():
                return x
            return njit(impl)
        tup1 = ('a', 'b', 'c', 12, 3j, ('f',))
        tup2 = (gen(1), gen(2), gen(3))
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

    def test_09(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            idx = 0
            for a in literal_unroll(tup1):
                if a == 'a':
                    acc += tup2[idx]
                elif a == 'b':
                    acc += tup2[idx]
                elif a == 'c':
                    acc += tup2[idx]
                idx += 1
            return (idx, acc)

        @njit
        def func1():
            return 1

        @njit
        def func2():
            return 2

        @njit
        def func3():
            return 3
        tup1 = ('a', 'b', 'c')
        tup2 = (1j, 1, 2)
        with self.assertRaises(errors.TypingError) as raises:
            foo(tup1, tup2)
        self.assertIn(_header_lead, str(raises.exception))

    def test_10(self):

        def dt(value):
            if value == 'apple':
                return 1
            elif value == 'orange':
                return 2
            elif value == 'banana':
                return 3
            elif value == 3390155550:
                return 1554098974 + value

        @overload(dt, inline='always')
        def ol_dt(li):
            if isinstance(li, types.StringLiteral):
                value = li.literal_value
                if value == 'apple':

                    def impl(li):
                        return 1
                elif value == 'orange':

                    def impl(li):
                        return 2
                elif value == 'banana':

                    def impl(li):
                        return 3
                return impl
            elif isinstance(li, types.IntegerLiteral):
                value = li.literal_value
                if value == 3390155550:

                    def impl(li):
                        return 1554098974 + value
                    return impl

        @njit
        def foo():
            acc = 0
            for t in literal_unroll(('apple', 'orange', 'banana', 3390155550)):
                acc += dt(t)
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_11(self):

        @njit
        def foo():
            x = []
            z = ('apple', 'orange', 'banana')
            for i in range(len(literal_unroll(z))):
                t = z[i]
                if t == 'apple':
                    x.append('0')
                elif t == 'orange':
                    x.append(t)
                elif t == 'banana':
                    x.append('2.0')
            return x
        self.assertEqual(foo(), foo.py_func())

    def test_11a(self):

        @njit
        def foo():
            x = typed.List()
            z = ('apple', 'orange', 'banana')
            for i in range(len(literal_unroll(z))):
                t = z[i]
                if t == 'apple':
                    x.append('0')
                elif t == 'orange':
                    x.append(t)
                elif t == 'banana':
                    x.append('2.0')
            return x
        self.assertEqual(foo(), foo.py_func())

    def test_12(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for i in literal_unroll(a):
                acc += i
                if acc.real < 26:
                    acc -= 1
                else:
                    for x in literal_unroll(a):
                        acc += x
                    break
            if a[0] < 23:
                acc += 2
            return acc
        f = 9
        k = f
        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_13(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for i in literal_unroll(a):
                acc += i
                if acc.real < 26:
                    acc -= 1
                else:
                    for x in literal_unroll(a):
                        for j in literal_unroll(a):
                            acc += j
                        acc += x
                for x in literal_unroll(a):
                    acc += x
            for x in literal_unroll(a):
                acc += x
            if a[0] < 23:
                acc += 2
            return acc
        f = 9
        k = f
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(2, k)
        self.assertIn('Nesting of literal_unroll is unsupported', str(raises.exception))

    def test_14(self):

        @njit
        def foo():
            x = (1, 2, 3, 4)
            acc = 0
            for a in literal_unroll(x):
                acc += a
            return a
        self.assertEqual(foo(), foo.py_func())

    def test_15(self):

        @njit
        def foo(x):
            acc = 0
            for a in literal_unroll(x):
                acc += len(a)
            return a
        n = 5
        tup = (np.ones((n,)), np.ones((n, n)), 'ABCDEFGHJI', (1, 2, 3), (1, 'foo', 2, 'bar'), {3, 4, 5, 6, 7})
        with self.assertRaises(errors.TypingError) as raises:
            foo(tup)
        self.assertIn('Cannot unify', str(raises.exception))

    def test_16(self):

        def dt(value):
            if value == 1000:
                return 'a'
            elif value == 2000:
                return 'b'
            elif value == 3000:
                return 'c'
            elif value == 4000:
                return 'd'

        @overload(dt, inline='always')
        def ol_dt(li):
            if isinstance(li, types.IntegerLiteral):
                value = li.literal_value
                if value == 1000:

                    def impl(li):
                        return 'a'
                elif value == 2000:

                    def impl(li):
                        return 'b'
                elif value == 3000:

                    def impl(li):
                        return 'c'
                elif value == 4000:

                    def impl(li):
                        return 'd'
                return impl

        @njit
        def foo():
            x = (1000, 2000, 3000, 4000)
            acc = ''
            for a in literal_unroll(x[:2]):
                acc += dt(a)
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_17(self):

        def dt(value):
            if value == 1000:
                return 'a'
            elif value == 2000:
                return 'b'
            elif value == 3000:
                return 'c'
            elif value == 4000:
                return 'd'
            elif value == 'f':
                return 'EFF'

        @overload(dt, inline='always')
        def ol_dt(li):
            if isinstance(li, types.IntegerLiteral):
                value = li.literal_value
                if value == 1000:

                    def impl(li):
                        return 'a'
                elif value == 2000:

                    def impl(li):
                        return 'b'
                elif value == 3000:

                    def impl(li):
                        return 'c'
                elif value == 4000:

                    def impl(li):
                        return 'd'
                return impl
            elif isinstance(li, types.StringLiteral):
                value = li.literal_value
                if value == 'f':

                    def impl(li):
                        return 'EFF'
                    return impl

        @njit
        def foo():
            x = (1000, 2000, 3000, 'f')
            acc = ''
            for a in literal_unroll(x[1:]):
                acc += dt(a)
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_18(self):

        @njit
        def foo():
            x = (1000, 2000, 3000, 4000, 5000, 6000)
            count = 0
            for a in literal_unroll(x[::-1]):
                count += 1
                if a < 3000:
                    break
            return count
        self.assertEqual(foo(), foo.py_func())

    def test_19(self):

        @njit
        def foo():
            acc = 0
            l1 = [1, 2, 3, 4]
            l2 = [10, 20]
            tup = (l1, l2)
            a1 = np.arange(20)
            a2 = np.ones(5, dtype=np.complex128)
            tup = (l1, a1, l2, a2)
            for t in literal_unroll(tup):
                acc += len(t)
            return acc
        self.assertEqual(foo(), foo.py_func())

    def test_20(self):

        @njit
        def foo():
            l = []
            a1 = np.arange(20)
            a2 = np.ones(5, dtype=np.complex128)
            tup = (a1, a2)
            for t in literal_unroll(tup):
                l.append(t.sum())
            return l
        self.assertEqual(foo(), foo.py_func())

    def test_21(self):

        @njit
        def foo(z):
            b = (23, 23.9, 6j, 8)

            def bar():
                acc = 0
                for j in literal_unroll(b):
                    acc += j
                return acc
            outer_acc = 0
            for x in (1, 2, 3, 4):
                outer_acc += bar() + x
            return outer_acc
        f = 9
        k = f
        self.assertEqual(foo(k), foo.py_func(k))

    def test_22(self):

        @njit
        def foo(z):
            a = (12, 12.7, 3j, 4, z, 2 * z, 'a')
            b = (23, 23.9, 6j, 8)

            def bar():
                acc = 0
                for j in literal_unroll(b):
                    acc += j
                return acc
            acc = 0
            for x in literal_unroll(a):
                acc += bar()
            return acc
        f = 9
        k = f
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(k)
        self.assertIn('Nesting of literal_unroll is unsupported', str(raises.exception))

    def test_23(self):

        @njit
        def foo(z):
            b = (23, 23.9, 6j, 8)

            def bar():
                acc = 0
                for j in literal_unroll(b):
                    acc += j
                return acc
            outer_acc = 0
            for x in literal_unroll(b):
                outer_acc += bar() + x
            return outer_acc
        f = 9
        k = f
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo(k)
        self.assertIn('Nesting of literal_unroll is unsupported', str(raises.exception))

    def test_24(self):

        @njit
        def foo():
            for x in literal_unroll('ABCDE'):
                print(x)
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        msg = 'argument should be a tuple or a list of constant values'
        self.assertIn(msg, str(raises.exception))

    def test_25(self):

        @njit
        def foo():
            val = literal_unroll(((1, 2, 3), (2j, 3j), [1, 2], 'xyz'))
            alias1 = val
            alias2 = alias1
            lens = []
            for x in alias2:
                lens.append(len(x))
            return lens
        self.assertEqual(foo(), foo.py_func())

    def test_26(self):

        @njit
        def foo(z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            count = 0
            untouched = 54
            read_only = 17
            mutated = np.empty((len(a),), dtype=np.complex128)
            for x in literal_unroll(a):
                acc += x
                mutated[count] = x
                count += 1
                escape = count + read_only
            return (escape, acc, untouched, read_only, mutated)
        f = 9
        k = f
        self.assertPreciseEqual(foo(k), foo.py_func(k))

    @skip_parfors_unsupported
    def test_27(self):

        @njit(parallel=True)
        def foo(z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for x in literal_unroll(a):
                for k in prange(10):
                    acc += 1
            return acc
        f = 9
        k = f
        self.assertEqual(foo(k), foo.py_func(k))

    @skip_parfors_unsupported
    def test_28(self):

        @njit(parallel=True)
        def foo(z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for x in literal_unroll(a):
                for k in prange(10):
                    acc += x
            return acc
        f = 9
        k = f
        np.testing.assert_allclose(foo(k), foo.py_func(k))

    @skip_parfors_unsupported
    def test_29(self):

        @njit(parallel=True)
        def foo(z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for k in prange(10):
                for x in literal_unroll(a):
                    acc += x
            return acc
        f = 9
        k = f
        self.assertEqual(foo(k), foo.py_func(k))

    def test_30(self):

        @njit
        def foo():
            const = 1234

            def bar(t):
                acc = 0
                a = (12, 12.7, 3j, 4)
                for x in literal_unroll(a):
                    acc += x + const
                return (acc, t)
            return [x for x in map(bar, (1, 2))]
        self.assertEqual(foo(), foo.py_func())

    def test_31(self):

        @njit
        def foo():
            const = 1234

            def bar(t):
                acc = 0
                a = (12, 12.7, 3j, 4)
                for x in literal_unroll(a):
                    acc += x + const
                return (acc, t)
            return [x for x in map(bar, (1, 2j))]
        with self.assertRaises(errors.TypingError) as raises:
            foo()
        self.assertIn(_header_lead, str(raises.exception))
        self.assertIn('zip', str(raises.exception))

    def test_32(self):

        @njit
        def gen(a):
            for x in literal_unroll(a):
                yield x

        @njit
        def foo():
            return [x for x in gen((1, 2.3, 4j))]
        self.assertEqual(foo(), foo.py_func())

    def test_33(self):

        @njit
        def consumer(func, arg):
            yield func(arg)

        def get(cons):

            @njit
            def foo():

                def gen(a):
                    for x in literal_unroll(a):
                        yield x
                return [next(x) for x in cons(gen, (1, 2.3, 4j))]
            return foo
        cfunc = get(consumer)
        pyfunc = get(consumer.py_func).py_func
        self.assertEqual(cfunc(), pyfunc())

    def test_34(self):

        @njit
        def foo():
            acc = 0
            l1 = [1, 2, 3, 4]
            l2 = [10, 20]
            if acc - 2 > 3:
                tup = (l1, l2)
            else:
                a1 = np.arange(20)
                a2 = np.ones(5, dtype=np.complex128)
                tup = (l1, a1, l2, a2)
            for t in literal_unroll(tup):
                acc += len(t)
            return acc
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        self.assertIn('Invalid use of', str(raises.exception))
        self.assertIn('found multiple definitions of variable', str(raises.exception))