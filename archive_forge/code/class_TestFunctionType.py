import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
class TestFunctionType(TestCase):
    """Test first-class functions in the context of a Numba jit compiled
    function.

    """

    def test_in__(self):
        """Function is passed in as an argument.
        """

        def a(i):
            return i + 1

        def foo(f):
            return 0
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_ctypes_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__, jit=jit_opts):
                    a_ = decor(a)
                    self.assertEqual(jit_(foo)(a_), foo(a))

    def test_in_call__(self):
        """Function is passed in as an argument and called.
        Also test different return values.
        """

        def a_i64(i):
            return i + 1234567

        def a_f64(i):
            return i + 1.5

        def a_str(i):
            return 'abc'

        def foo(f):
            return f(123)
        for f, sig in [(a_i64, int64(int64)), (a_f64, float64(int64))]:
            for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
                for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                    jit_ = jit(**jit_opts)
                    with self.subTest(sig=sig, decor=decor.__name__, jit=jit_opts):
                        f_ = decor(f)
                        self.assertEqual(jit_(foo)(f_), foo(f))

    def test_in_call_out(self):
        """Function is passed in as an argument, called, and returned.
        """

        def a(i):
            return i + 1

        def foo(f):
            f(123)
            return f
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    r1 = jit_(foo)(a_).pyfunc
                    r2 = foo(a)
                    self.assertEqual(r1, r2)

    def test_in_seq_call(self):
        """Functions are passed in as arguments, used as tuple items, and
        called.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(f, g):
            r = 0
            for f_ in (f, g):
                r = r + f_(r)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_), foo(a, b))

    def test_in_ns_seq_call(self):
        """Functions are passed in as an argument and via namespace scoping
        (mixed pathways), used as tuple items, and called.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def mkfoo(b_):

            def foo(f):
                r = 0
                for f_ in (f, b_):
                    r = r + f_(r)
                return r
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(mkfoo(b_))(a_), mkfoo(b)(a))

    def test_ns_call(self):
        """Function is passed in via namespace scoping and called.

        """

        def a(i):
            return i + 1

        def mkfoo(a_):

            def foo():
                return a_(123)
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))(), mkfoo(a)())

    def test_ns_out(self):
        """Function is passed in via namespace scoping and returned.

        """

        def a(i):
            return i + 1

        def mkfoo(a_):

            def foo():
                return a_
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_ns_call_out(self):
        """Function is passed in via namespace scoping, called, and then
        returned.

        """

        def a(i):
            return i + 1

        def mkfoo(a_):

            def foo():
                a_(123)
                return a_
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(jit_(mkfoo(a_))().pyfunc, mkfoo(a)())

    def test_in_overload(self):
        """Function is passed in as an argument and called with different
        argument types.

        """

        def a(i):
            return i + 1

        def foo(f):
            r1 = f(123)
            r2 = f(123.45)
            return (r1, r2)
        for decor in [njit_func]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(foo)(a_), foo(a))

    def test_ns_overload(self):
        """Function is passed in via namespace scoping and called with
        different argument types.

        """

        def a(i):
            return i + 1

        def mkfoo(a_):

            def foo():
                r1 = a_(123)
                r2 = a_(123.45)
                return (r1, r2)
            return foo
        for decor in [njit_func]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    self.assertEqual(jit_(mkfoo(a_))(), mkfoo(a)())

    def test_in_choose(self):
        """Functions are passed in as arguments and called conditionally.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                r = a(1)
            else:
                r = b(2)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True), foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False), foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True), foo(a, b, False))

    def test_ns_choose(self):
        """Functions are passed in via namespace scoping and called
        conditionally.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def mkfoo(a_, b_):

            def foo(choose_left):
                if choose_left:
                    r = a_(1)
                else:
                    r = b_(2)
                return r
            return foo
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(mkfoo(a_, b_))(True), mkfoo(a, b)(True))
                    self.assertEqual(jit_(mkfoo(a_, b_))(False), mkfoo(a, b)(False))
                    self.assertNotEqual(jit_(mkfoo(a_, b_))(True), mkfoo(a, b)(False))

    def test_in_choose_out(self):
        """Functions are passed in as arguments and returned conditionally.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                return a
            else:
                return b
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False).pyfunc, foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True).pyfunc, foo(a, b, False))

    def test_in_choose_func_value(self):
        """Functions are passed in as arguments, selected conditionally and
        called.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(a, b, choose_left):
            if choose_left:
                f = a
            else:
                f = b
            return f(1)
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), njit_func, mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)(a_, b_, True), foo(a, b, True))
                    self.assertEqual(jit_(foo)(a_, b_, False), foo(a, b, False))
                    self.assertNotEqual(jit_(foo)(a_, b_, True), foo(a, b, False))

    def test_in_pick_func_call(self):
        """Functions are passed in as items of tuple argument, retrieved via
        indexing, and called.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(funcs, i):
            f = funcs[i]
            r = f(123)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)((a_, b_), 0), foo((a, b), 0))
                    self.assertEqual(jit_(foo)((a_, b_), 1), foo((a, b), 1))
                    self.assertNotEqual(jit_(foo)((a_, b_), 0), foo((a, b), 1))

    def test_in_iter_func_call(self):
        """Functions are passed in as items of tuple argument, retrieved via
        indexing, and called within a variable for-loop.

        """

        def a(i):
            return i + 1

        def b(i):
            return i + 2

        def foo(funcs, n):
            r = 0
            for i in range(n):
                f = funcs[i]
                r = r + f(r)
            return r
        sig = int64(int64)
        for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
            for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
                jit_ = jit(**jit_opts)
                with self.subTest(decor=decor.__name__):
                    a_ = decor(a)
                    b_ = decor(b)
                    self.assertEqual(jit_(foo)((a_, b_), 2), foo((a, b), 2))

    def test_experimental_feature_warning(self):

        @jit(nopython=True)
        def more(x):
            return x + 1

        @jit(nopython=True)
        def less(x):
            return x - 1

        @jit(nopython=True)
        def foo(sel, x):
            fn = more if sel else less
            return fn(x)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            res = foo(True, 10)
        self.assertEqual(res, 11)
        self.assertEqual(foo(False, 10), 9)
        self.assertGreaterEqual(len(ws), 1)
        pat = 'First-class function type feature is experimental'
        for w in ws:
            if pat in str(w.message):
                break
        else:
            self.fail('missing warning')