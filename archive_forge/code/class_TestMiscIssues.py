import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class TestMiscIssues(TestCase):
    """Test issues of using first-class functions in the context of Numba
    jit compiled functions.

    """

    def test_issue_3405_using_cfunc(self):

        @cfunc('int64()')
        def a():
            return 2

        @cfunc('int64()')
        def b():
            return 3

        def g(arg):
            if arg:
                f = a
            else:
                f = b
            return f()
        self.assertEqual(jit(nopython=True)(g)(True), 2)
        self.assertEqual(jit(nopython=True)(g)(False), 3)

    def test_issue_3405_using_njit(self):

        @jit(nopython=True)
        def a():
            return 2

        @jit(nopython=True)
        def b():
            return 3

        def g(arg):
            if not arg:
                f = b
            else:
                f = a
            return f()
        self.assertEqual(jit(nopython=True)(g)(True), 2)
        self.assertEqual(jit(nopython=True)(g)(False), 3)

    def test_pr4967_example(self):

        @cfunc('int64(int64)')
        def a(i):
            return i + 1

        @cfunc('int64(int64)')
        def b(i):
            return i + 2

        @jit(nopython=True)
        def foo(f, g):
            i = f(2)
            seq = (f, g)
            for fun in seq:
                i += fun(i)
            return i
        a_ = a._pyfunc
        b_ = b._pyfunc
        self.assertEqual(foo(a, b), a_(2) + a_(a_(2)) + b_(a_(2) + a_(a_(2))))

    def test_pr4967_array(self):
        import numpy as np

        @cfunc('intp(intp[:], float64[:])')
        def foo1(x, y):
            return x[0] + y[0]

        @cfunc('intp(intp[:], float64[:])')
        def foo2(x, y):
            return x[0] - y[0]

        def bar(fx, fy, i):
            a = np.array([10], dtype=np.intp)
            b = np.array([12], dtype=np.float64)
            if i == 0:
                f = fx
            elif i == 1:
                f = fy
            else:
                return
            return f(a, b)
        r = jit(nopython=True, no_cfunc_wrapper=True)(bar)(foo1, foo2, 0)
        self.assertEqual(r, bar(foo1, foo2, 0))
        self.assertNotEqual(r, bar(foo1, foo2, 1))

    def test_reference_example(self):
        import numba

        @numba.njit
        def composition(funcs, x):
            r = x
            for f in funcs[::-1]:
                r = f(r)
            return r

        @numba.cfunc('double(double)')
        def a(x):
            return x + 1.0

        @numba.njit()
        def b(x):
            return x * x
        r = composition((a, b, b, a), 0.5)
        self.assertEqual(r, (0.5 + 1.0) ** 4 + 1.0)
        r = composition((b, a, b, b, a), 0.5)
        self.assertEqual(r, ((0.5 + 1.0) ** 4 + 1.0) ** 2)

    def test_apply_function_in_function(self):

        def foo(f, f_inner):
            return f(f_inner)

        @cfunc('int64(float64)')
        def f_inner(i):
            return int64(i * 3)

        @cfunc(int64(types.FunctionType(f_inner._sig)))
        def f(f_inner):
            return f_inner(123.4)
        self.assertEqual(jit(nopython=True)(foo)(f, f_inner), foo(f._pyfunc, f_inner._pyfunc))

    def test_function_with_none_argument(self):

        @cfunc(int64(types.none))
        def a(i):
            return 1

        @jit(nopython=True)
        def foo(f):
            return f(None)
        self.assertEqual(foo(a), 1)

    def test_constant_functions(self):

        @jit(nopython=True)
        def a():
            return 123

        @jit(nopython=True)
        def b():
            return 456

        @jit(nopython=True)
        def foo():
            return a() + b()
        r = foo()
        if r != 123 + 456:
            print(foo.overloads[()].library.get_llvm_str())
        self.assertEqual(r, 123 + 456)

    def test_generators(self):

        @jit(forceobj=True)
        def gen(xs):
            for x in xs:
                x += 1
                yield x

        @jit(forceobj=True)
        def con(gen_fn, xs):
            return [it for it in gen_fn(xs)]
        self.assertEqual(con(gen, (1, 2, 3)), [2, 3, 4])

        @jit(nopython=True)
        def gen_(xs):
            for x in xs:
                x += 1
                yield x
        self.assertEqual(con(gen_, (1, 2, 3)), [2, 3, 4])

    def test_jit_support(self):

        @jit(nopython=True)
        def foo(f, x):
            return f(x)

        @jit()
        def a(x):
            return x + 1

        @jit()
        def a2(x):
            return x - 1

        @jit()
        def b(x):
            return x + 1.5
        self.assertEqual(foo(a, 1), 2)
        a2(5)
        self.assertEqual(foo(a2, 2), 1)
        self.assertEqual(foo(a2, 3), 2)
        self.assertEqual(foo(a, 2), 3)
        self.assertEqual(foo(a, 1.5), 2.5)
        self.assertEqual(foo(a2, 1), 0)
        self.assertEqual(foo(a, 2.5), 3.5)
        self.assertEqual(foo(b, 1.5), 3.0)
        self.assertEqual(foo(b, 1), 2.5)

    def test_signature_mismatch(self):

        @jit(nopython=True)
        def f1(x):
            return x

        @jit(nopython=True)
        def f2(x):
            return x

        @jit(nopython=True)
        def foo(disp1, disp2, sel):
            if sel == 1:
                fn = disp1
            else:
                fn = disp2
            return (fn([1]), fn(2))
        with self.assertRaises(errors.UnsupportedError) as cm:
            foo(f1, f2, sel=1)
        self.assertRegex(str(cm.exception), 'mismatch of function types:')
        self.assertEqual(foo(f1, f1, sel=1), ([1], 2))

    def test_unique_dispatcher(self):

        def foo_template(funcs, x):
            r = x
            for f in funcs:
                r = f(r)
            return r
        a = jit(nopython=True)(lambda x: x + 1)
        b = jit(nopython=True)(lambda x: x + 2)
        foo = jit(nopython=True)(foo_template)
        a(0)
        a.disable_compile()
        r = foo((a, b), 0)
        self.assertEqual(r, 3)
        self.assertEqual(foo.signatures[0][0].dtype.is_precise(), True)

    def test_zero_address(self):
        sig = int64()

        @cfunc(sig)
        def test():
            return 123

        class Good(types.WrapperAddressProtocol):
            """A first-class function type with valid address.
            """

            def __wrapper_address__(self):
                return test.address

            def signature(self):
                return sig

        class Bad(types.WrapperAddressProtocol):
            """A first-class function type with invalid 0 address.
            """

            def __wrapper_address__(self):
                return 0

            def signature(self):
                return sig

        class BadToGood(types.WrapperAddressProtocol):
            """A first-class function type with invalid address that is
            recovered to a valid address.
            """
            counter = -1

            def __wrapper_address__(self):
                self.counter += 1
                return test.address * min(1, self.counter)

            def signature(self):
                return sig
        good = Good()
        bad = Bad()
        bad2good = BadToGood()

        @jit(int64(sig.as_type()))
        def foo(func):
            return func()

        @jit(int64())
        def foo_good():
            return good()

        @jit(int64())
        def foo_bad():
            return bad()

        @jit(int64())
        def foo_bad2good():
            return bad2good()
        self.assertEqual(foo(good), 123)
        self.assertEqual(foo_good(), 123)
        with self.assertRaises(ValueError) as cm:
            foo(bad)
        self.assertRegex(str(cm.exception), 'wrapper address of <.*> instance must be a positive')
        with self.assertRaises(RuntimeError) as cm:
            foo_bad()
        self.assertRegex(str(cm.exception), '.* function address is null')
        self.assertEqual(foo_bad2good(), 123)

    def test_issue_5470(self):

        @njit()
        def foo1():
            return 10

        @njit()
        def foo2():
            return 20
        formulae_foo = (foo1, foo1)

        @njit()
        def bar_scalar(f1, f2):
            return f1() + f2()

        @njit()
        def bar():
            return bar_scalar(*formulae_foo)
        self.assertEqual(bar(), 20)
        formulae_foo = (foo1, foo2)

        @njit()
        def bar():
            return bar_scalar(*formulae_foo)
        self.assertEqual(bar(), 30)

    def test_issue_5540(self):

        @njit(types.int64(types.int64))
        def foo(x):
            return x + 1

        @njit
        def bar_bad(foos):
            f = foos[0]
            return f(x=1)

        @njit
        def bar_good(foos):
            f = foos[0]
            return f(1)
        self.assertEqual(bar_good((foo,)), 2)
        with self.assertRaises((errors.UnsupportedError, errors.TypingError)) as cm:
            bar_bad((foo,))
        self.assertRegex(str(cm.exception), '.*first-class function call cannot use keyword arguments')

    def test_issue_5615(self):

        @njit
        def foo1(x):
            return x + 1

        @njit
        def foo2(x):
            return x + 2

        @njit
        def bar(fcs):
            x = 0
            a = 10
            i, j = fcs[0]
            x += i(j(a))
            for t in literal_unroll(fcs):
                i, j = t
                x += i(j(a))
            return x
        tup = ((foo1, foo2), (foo2, foo1))
        self.assertEqual(bar(tup), 39)

    def test_issue_5685(self):

        @njit
        def foo1():
            return 1

        @njit
        def foo2(x):
            return x + 1

        @njit
        def foo3(x):
            return x + 2

        @njit
        def bar(fcs):
            r = 0
            for pair in literal_unroll(fcs):
                f1, f2 = pair
                r += f1() + f2(2)
            return r
        self.assertEqual(bar(((foo1, foo2),)), 4)
        self.assertEqual(bar(((foo1, foo2), (foo1, foo3))), 9)