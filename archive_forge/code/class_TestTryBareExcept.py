import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
class TestTryBareExcept(TestCase):
    """Test the following pattern:

        try:
            <body>
        except:
            <handling>
    """

    def test_try_inner_raise(self):

        @njit
        def inner(x):
            if x:
                raise MyError

        @njit
        def udt(x):
            try:
                inner(x)
                return 'not raised'
            except:
                return 'caught'
        self.assertEqual(udt(False), 'not raised')
        self.assertEqual(udt(True), 'caught')

    def test_try_state_reset(self):

        @njit
        def inner(x):
            if x == 1:
                raise MyError('one')
            elif x == 2:
                raise MyError('two')

        @njit
        def udt(x):
            try:
                inner(x)
                res = 'not raised'
            except:
                res = 'caught'
            if x == 0:
                inner(2)
            return res
        with self.assertRaises(MyError) as raises:
            udt(0)
        self.assertEqual(str(raises.exception), 'two')
        self.assertEqual(udt(1), 'caught')
        self.assertEqual(udt(-1), 'not raised')

    def _multi_inner(self):

        @njit
        def inner(x):
            if x == 1:
                print('call_one')
                raise MyError('one')
            elif x == 2:
                print('call_two')
                raise MyError('two')
            elif x == 3:
                print('call_three')
                raise MyError('three')
            else:
                print('call_other')
        return inner

    def test_nested_try(self):
        inner = self._multi_inner()

        @njit
        def udt(x, y, z):
            try:
                try:
                    print('A')
                    inner(x)
                    print('B')
                except:
                    print('C')
                    inner(y)
                    print('D')
            except:
                print('E')
                inner(z)
                print('F')
        with self.assertRaises(MyError) as raises:
            with captured_stdout() as stdout:
                udt(1, 2, 3)
        self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_two', 'E', 'call_three'])
        self.assertEqual(str(raises.exception), 'three')
        with captured_stdout() as stdout:
            udt(1, 0, 3)
        self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_other', 'D'])
        with captured_stdout() as stdout:
            udt(1, 2, 0)
        self.assertEqual(stdout.getvalue().split(), ['A', 'call_one', 'C', 'call_two', 'E', 'call_other', 'F'])

    def test_loop_in_try(self):
        inner = self._multi_inner()

        @njit
        def udt(x, n):
            try:
                print('A')
                for i in range(n):
                    print(i)
                    if i == x:
                        inner(i)
            except:
                print('B')
            return i
        with captured_stdout() as stdout:
            res = udt(3, 5)
        self.assertEqual(stdout.getvalue().split(), ['A', '0', '1', '2', '3', 'call_three', 'B'])
        self.assertEqual(res, 3)
        with captured_stdout() as stdout:
            res = udt(1, 3)
        self.assertEqual(stdout.getvalue().split(), ['A', '0', '1', 'call_one', 'B'])
        self.assertEqual(res, 1)
        with captured_stdout() as stdout:
            res = udt(0, 3)
        self.assertEqual(stdout.getvalue().split(), ['A', '0', 'call_other', '1', '2'])
        self.assertEqual(res, 2)

    def test_raise_in_try(self):

        @njit
        def udt(x):
            try:
                print('A')
                if x:
                    raise MyError('my_error')
                print('B')
            except:
                print('C')
                return 321
            return 123
        with captured_stdout() as stdout:
            res = udt(True)
        self.assertEqual(stdout.getvalue().split(), ['A', 'C'])
        self.assertEqual(res, 321)
        with captured_stdout() as stdout:
            res = udt(False)
        self.assertEqual(stdout.getvalue().split(), ['A', 'B'])
        self.assertEqual(res, 123)

    def test_recursion(self):

        @njit
        def foo(x):
            if x > 0:
                try:
                    foo(x - 1)
                except:
                    print('CAUGHT')
                    return 12
            if x == 1:
                raise ValueError('exception')
        with captured_stdout() as stdout:
            res = foo(10)
        self.assertIsNone(res)
        self.assertEqual(stdout.getvalue().split(), ['CAUGHT'])

    def test_yield(self):

        @njit
        def foo(x):
            if x > 0:
                try:
                    yield 7
                    raise ValueError('exception')
                except Exception:
                    print('CAUGHT')

        @njit
        def bar(z):
            return next(foo(z))
        with captured_stdout() as stdout:
            res = bar(10)
        self.assertEqual(res, 7)
        self.assertEqual(stdout.getvalue().split(), [])

    def test_closure2(self):

        @njit
        def foo(x):

            def bar():
                try:
                    raise ValueError('exception')
                except:
                    print('CAUGHT')
                    return 12
            bar()
        with captured_stdout() as stdout:
            foo(10)
        self.assertEqual(stdout.getvalue().split(), ['CAUGHT'])

    def test_closure3(self):

        @njit
        def foo(x):

            def bar(z):
                try:
                    raise ValueError('exception')
                except:
                    print('CAUGHT')
                    return z
            return [x for x in map(bar, [1, 2, 3])]
        with captured_stdout() as stdout:
            res = foo(10)
        self.assertEqual(res, [1, 2, 3])
        self.assertEqual(stdout.getvalue().split(), ['CAUGHT'] * 3)

    def test_closure4(self):

        @njit
        def foo(x):

            def bar(z):
                if z < 0:
                    raise ValueError('exception')
                return z
            try:
                return [x for x in map(bar, [1, 2, 3, x])]
            except:
                print('CAUGHT')
        with captured_stdout() as stdout:
            res = foo(-1)
        self.assertEqual(stdout.getvalue().strip(), 'CAUGHT')
        self.assertIsNone(res)
        with captured_stdout() as stdout:
            res = foo(4)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(res, [1, 2, 3, 4])

    @skip_unless_scipy
    def test_real_problem(self):

        @njit
        def foo():
            a = np.zeros((4, 4))
            try:
                chol = np.linalg.cholesky(a)
            except:
                print('CAUGHT')
                return chol
        with captured_stdout() as stdout:
            foo()
        self.assertEqual(stdout.getvalue().split(), ['CAUGHT'])

    def test_for_loop(self):

        @njit
        def foo(n):
            for i in range(n):
                try:
                    if i > 5:
                        raise ValueError
                except:
                    print('CAUGHT')
            else:
                try:
                    try:
                        try:
                            if i > 5:
                                raise ValueError
                        except:
                            print('CAUGHT1')
                            raise ValueError
                    except:
                        print('CAUGHT2')
                        raise ValueError
                except:
                    print('CAUGHT3')
        with captured_stdout() as stdout:
            foo(10)
        self.assertEqual(stdout.getvalue().split(), ['CAUGHT'] * 4 + ['CAUGHT%s' % i for i in range(1, 4)])

    def test_try_pass(self):

        @njit
        def foo(x):
            try:
                pass
            except:
                pass
            return x
        res = foo(123)
        self.assertEqual(res, 123)

    def test_try_except_reraise(self):

        @njit
        def udt():
            try:
                raise ValueError('ERROR')
            except:
                raise
        with self.assertRaises(UnsupportedError) as raises:
            udt()
        self.assertIn('The re-raising of an exception is not yet supported.', str(raises.exception))