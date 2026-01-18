import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
class TestJitClassOverloads(MemoryLeakMixin, TestCase):

    class PyList:

        def __init__(self):
            self.x = [0]

        def append(self, y):
            self.x.append(y)

        def clear(self):
            self.x.clear()

        def __abs__(self):
            return len(self.x) * 7

        def __bool__(self):
            return len(self.x) % 3 != 0

        def __complex__(self):
            c = complex(2)
            if self.x:
                c += self.x[0]
            return c

        def __contains__(self, y):
            return y in self.x

        def __float__(self):
            f = 3.1415
            if self.x:
                f += self.x[0]
            return f

        def __int__(self):
            i = 5
            if self.x:
                i += self.x[0]
            return i

        def __len__(self):
            return len(self.x) + 1

        def __str__(self):
            if len(self.x) == 0:
                return 'PyList empty'
            else:
                return 'PyList non-empty'

    @staticmethod
    def get_int_wrapper():

        @jitclass([('x', types.intp)])
        class IntWrapper:

            def __init__(self, value):
                self.x = value

            def __eq__(self, other):
                return self.x == other.x

            def __hash__(self):
                return self.x

            def __lshift__(self, other):
                return IntWrapper(self.x << other.x)

            def __rshift__(self, other):
                return IntWrapper(self.x >> other.x)

            def __and__(self, other):
                return IntWrapper(self.x & other.x)

            def __or__(self, other):
                return IntWrapper(self.x | other.x)

            def __xor__(self, other):
                return IntWrapper(self.x ^ other.x)
        return IntWrapper

    @staticmethod
    def get_float_wrapper():

        @jitclass([('x', types.float64)])
        class FloatWrapper:

            def __init__(self, value):
                self.x = value

            def __eq__(self, other):
                return self.x == other.x

            def __hash__(self):
                return self.x

            def __ge__(self, other):
                return self.x >= other.x

            def __gt__(self, other):
                return self.x > other.x

            def __le__(self, other):
                return self.x <= other.x

            def __lt__(self, other):
                return self.x < other.x

            def __add__(self, other):
                return FloatWrapper(self.x + other.x)

            def __floordiv__(self, other):
                return FloatWrapper(self.x // other.x)

            def __mod__(self, other):
                return FloatWrapper(self.x % other.x)

            def __mul__(self, other):
                return FloatWrapper(self.x * other.x)

            def __neg__(self, other):
                return FloatWrapper(-self.x)

            def __pos__(self, other):
                return FloatWrapper(+self.x)

            def __pow__(self, other):
                return FloatWrapper(self.x ** other.x)

            def __sub__(self, other):
                return FloatWrapper(self.x - other.x)

            def __truediv__(self, other):
                return FloatWrapper(self.x / other.x)
        return FloatWrapper

    def assertSame(self, first, second, msg=None):
        self.assertEqual(type(first), type(second), msg=msg)
        self.assertEqual(first, second, msg=msg)

    def test_overloads(self):
        JitList = jitclass({'x': types.List(types.intp)})(self.PyList)
        py_funcs = [lambda x: abs(x), lambda x: x.__abs__(), lambda x: bool(x), lambda x: x.__bool__(), lambda x: complex(x), lambda x: x.__complex__(), lambda x: 0 in x, lambda x: x.__contains__(0), lambda x: float(x), lambda x: x.__float__(), lambda x: int(x), lambda x: x.__int__(), lambda x: len(x), lambda x: x.__len__(), lambda x: str(x), lambda x: x.__str__(), lambda x: 1 if x else 0]
        jit_funcs = [njit(f) for f in py_funcs]
        py_list = self.PyList()
        jit_list = JitList()
        for py_f, jit_f in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.append(2)
        jit_list.append(2)
        for py_f, jit_f in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.append(-5)
        jit_list.append(-5)
        for py_f, jit_f in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.clear()
        jit_list.clear()
        for py_f, jit_f in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))

    def test_bool_fallback(self):

        def py_b(x):
            return bool(x)
        jit_b = njit(py_b)

        @jitclass([('x', types.List(types.intp))])
        class LenClass:

            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x) % 4

            def append(self, y):
                self.x.append(y)

            def pop(self):
                self.x.pop(0)
        obj = LenClass([1, 2, 3])
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))
        obj.append(4)
        self.assertFalse(py_b(obj))
        self.assertFalse(jit_b(obj))
        obj.pop()
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))

        @jitclass([('y', types.float64)])
        class NormalClass:

            def __init__(self, y):
                self.y = y
        obj = NormalClass(0)
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))

    def test_numeric_fallback(self):

        def py_c(x):
            return complex(x)

        def py_f(x):
            return float(x)

        def py_i(x):
            return int(x)
        jit_c = njit(py_c)
        jit_f = njit(py_f)
        jit_i = njit(py_i)

        @jitclass([])
        class FloatClass:

            def __init__(self):
                pass

            def __float__(self):
                return 3.1415
        obj = FloatClass()
        self.assertSame(py_c(obj), complex(3.1415))
        self.assertSame(jit_c(obj), complex(3.1415))
        self.assertSame(py_f(obj), 3.1415)
        self.assertSame(jit_f(obj), 3.1415)
        with self.assertRaises(TypeError) as e:
            py_i(obj)
        self.assertIn('int', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_i(obj)
        self.assertIn('int', str(e.exception))

        @jitclass([])
        class IntClass:

            def __init__(self):
                pass

            def __int__(self):
                return 7
        obj = IntClass()
        self.assertSame(py_i(obj), 7)
        self.assertSame(jit_i(obj), 7)
        with self.assertRaises(TypeError) as e:
            py_c(obj)
        self.assertIn('complex', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_c(obj)
        self.assertIn('complex', str(e.exception))
        with self.assertRaises(TypeError) as e:
            py_f(obj)
        self.assertIn('float', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_f(obj)
        self.assertIn('float', str(e.exception))

        @jitclass([])
        class IndexClass:

            def __init__(self):
                pass

            def __index__(self):
                return 1
        obj = IndexClass()
        self.assertSame(py_c(obj), complex(1))
        self.assertSame(jit_c(obj), complex(1))
        self.assertSame(py_f(obj), 1.0)
        self.assertSame(jit_f(obj), 1.0)
        self.assertSame(py_i(obj), 1)
        self.assertSame(jit_i(obj), 1)

        @jitclass([])
        class FloatIntIndexClass:

            def __init__(self):
                pass

            def __float__(self):
                return 3.1415

            def __int__(self):
                return 7

            def __index__(self):
                return 1
        obj = FloatIntIndexClass()
        self.assertSame(py_c(obj), complex(3.1415))
        self.assertSame(jit_c(obj), complex(3.1415))
        self.assertSame(py_f(obj), 3.1415)
        self.assertSame(jit_f(obj), 3.1415)
        self.assertSame(py_i(obj), 7)
        self.assertSame(jit_i(obj), 7)

    def test_arithmetic_logical(self):
        IntWrapper = self.get_int_wrapper()
        FloatWrapper = self.get_float_wrapper()
        float_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x >= y, lambda x, y: x > y, lambda x, y: x <= y, lambda x, y: x < y, lambda x, y: x + y, lambda x, y: x // y, lambda x, y: x % y, lambda x, y: x * y, lambda x, y: x ** y, lambda x, y: x - y, lambda x, y: x / y]
        int_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x << y, lambda x, y: x >> y, lambda x, y: x & y, lambda x, y: x | y, lambda x, y: x ^ y]
        test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]

        def unwrap(value):
            return getattr(value, 'x', value)
        for jit_f, (x, y) in itertools.product(map(njit, float_py_funcs), test_values):
            py_f = jit_f.py_func
            expected = py_f(x, y)
            jit_x = FloatWrapper(x)
            jit_y = FloatWrapper(y)
            check = self.assertEqual if type(expected) is not float else self.assertAlmostEqual
            check(expected, jit_f(x, y))
            check(expected, unwrap(py_f(jit_x, jit_y)))
            check(expected, unwrap(jit_f(jit_x, jit_y)))
        for jit_f, (x, y) in itertools.product(map(njit, int_py_funcs), test_values):
            py_f = jit_f.py_func
            x, y = (int(x), int(y))
            expected = py_f(x, y)
            jit_x = IntWrapper(x)
            jit_y = IntWrapper(y)
            self.assertEqual(expected, jit_f(x, y))
            self.assertEqual(expected, unwrap(py_f(jit_x, jit_y)))
            self.assertEqual(expected, unwrap(jit_f(jit_x, jit_y)))

    def test_arithmetic_logical_inplace(self):
        JitIntWrapper = self.get_int_wrapper()
        JitFloatWrapper = self.get_float_wrapper()
        PyIntWrapper = JitIntWrapper.mro()[1]
        PyFloatWrapper = JitFloatWrapper.mro()[1]

        @jitclass([('x', types.intp)])
        class JitIntUpdateWrapper(PyIntWrapper):

            def __init__(self, value):
                self.x = value

            def __ilshift__(self, other):
                return JitIntUpdateWrapper(self.x << other.x)

            def __irshift__(self, other):
                return JitIntUpdateWrapper(self.x >> other.x)

            def __iand__(self, other):
                return JitIntUpdateWrapper(self.x & other.x)

            def __ior__(self, other):
                return JitIntUpdateWrapper(self.x | other.x)

            def __ixor__(self, other):
                return JitIntUpdateWrapper(self.x ^ other.x)

        @jitclass({'x': types.float64})
        class JitFloatUpdateWrapper(PyFloatWrapper):

            def __init__(self, value):
                self.x = value

            def __iadd__(self, other):
                return JitFloatUpdateWrapper(self.x + 2.718 * other.x)

            def __ifloordiv__(self, other):
                return JitFloatUpdateWrapper(self.x * 2.718 // other.x)

            def __imod__(self, other):
                return JitFloatUpdateWrapper(self.x % (other.x + 1))

            def __imul__(self, other):
                return JitFloatUpdateWrapper(self.x * other.x + 1)

            def __ipow__(self, other):
                return JitFloatUpdateWrapper(self.x ** other.x + 1)

            def __isub__(self, other):
                return JitFloatUpdateWrapper(self.x - 3.1415 * other.x)

            def __itruediv__(self, other):
                return JitFloatUpdateWrapper((self.x + 1) / other.x)
        PyIntUpdateWrapper = JitIntUpdateWrapper.mro()[1]
        PyFloatUpdateWrapper = JitFloatUpdateWrapper.mro()[1]

        def get_update_func(op):
            template = f'\ndef f(x, y):\n    x {op}= y\n    return x\n'
            namespace = {}
            exec(template, namespace)
            return namespace['f']
        float_py_funcs = [get_update_func(op) for op in ['+', '//', '%', '*', '**', '-', '/']]
        int_py_funcs = [get_update_func(op) for op in ['<<', '>>', '&', '|', '^']]
        test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]
        for jit_f, (py_cls, jit_cls), (x, y) in itertools.product(map(njit, float_py_funcs), [(PyFloatWrapper, JitFloatWrapper), (PyFloatUpdateWrapper, JitFloatUpdateWrapper)], test_values):
            py_f = jit_f.py_func
            expected = py_f(py_cls(x), py_cls(y)).x
            self.assertAlmostEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
            self.assertAlmostEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)
        for jit_f, (py_cls, jit_cls), (x, y) in itertools.product(map(njit, int_py_funcs), [(PyIntWrapper, JitIntWrapper), (PyIntUpdateWrapper, JitIntUpdateWrapper)], test_values):
            x, y = (int(x), int(y))
            py_f = jit_f.py_func
            expected = py_f(py_cls(x), py_cls(y)).x
            self.assertEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
            self.assertEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)

    def test_hash_eq_ne(self):

        class HashEqTest:
            x: int

            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return self.x % 10

            def __eq__(self, o):
                return (self.x - o.x) % 20 == 0

        class HashEqNeTest(HashEqTest):

            def __ne__(self, o):
                return (self.x - o.x) % 20 > 1

        def py_hash(x):
            return hash(x)

        def py_eq(x, y):
            return x == y

        def py_ne(x, y):
            return x != y

        def identity_decorator(f):
            return f
        comparisons = [(0, 1), (2, 22), (7, 10), (3, 3)]
        for base_cls, use_jit in itertools.product([HashEqTest, HashEqNeTest], [False, True]):
            decorator = njit if use_jit else identity_decorator
            hash_func = decorator(py_hash)
            eq_func = decorator(py_eq)
            ne_func = decorator(py_ne)
            jit_cls = jitclass(base_cls)
            for v in [0, 2, 10, 24, -8]:
                self.assertEqual(hash_func(jit_cls(v)), v % 10)
            for x, y in comparisons:
                self.assertEqual(eq_func(jit_cls(x), jit_cls(y)), base_cls(x) == base_cls(y))
                self.assertEqual(ne_func(jit_cls(x), jit_cls(y)), base_cls(x) != base_cls(y))

    def test_bool_fallback_len(self):

        class NoBoolHasLen:

            def __init__(self, val):
                self.val = val

            def __len__(self):
                return self.val

            def get_bool(self):
                return bool(self)
        py_class = NoBoolHasLen
        jitted_class = jitclass([('val', types.int64)])(py_class)
        py_class_0_bool = py_class(0).get_bool()
        py_class_2_bool = py_class(2).get_bool()
        jitted_class_0_bool = jitted_class(0).get_bool()
        jitted_class_2_bool = jitted_class(2).get_bool()
        self.assertEqual(py_class_0_bool, jitted_class_0_bool)
        self.assertEqual(py_class_2_bool, jitted_class_2_bool)
        self.assertEqual(type(py_class_0_bool), type(jitted_class_0_bool))
        self.assertEqual(type(py_class_2_bool), type(jitted_class_2_bool))

    def test_bool_fallback_default(self):

        class NoBoolNoLen:

            def __init__(self):
                pass

            def get_bool(self):
                return bool(self)
        py_class = NoBoolNoLen
        jitted_class = jitclass([])(py_class)
        py_class_bool = py_class().get_bool()
        jitted_class_bool = jitted_class().get_bool()
        self.assertEqual(py_class_bool, jitted_class_bool)
        self.assertEqual(type(py_class_bool), type(jitted_class_bool))

    def test_operator_reflection(self):

        class OperatorsDefined:

            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

            def __le__(self, other):
                return self.x <= other.x

            def __lt__(self, other):
                return self.x < other.x

            def __ge__(self, other):
                return self.x >= other.x

            def __gt__(self, other):
                return self.x > other.x

        class NoOperatorsDefined:

            def __init__(self, x):
                self.x = x
        spec = [('x', types.int32)]
        JitOperatorsDefined = jitclass(spec)(OperatorsDefined)
        JitNoOperatorsDefined = jitclass(spec)(NoOperatorsDefined)
        py_ops_defined = OperatorsDefined(2)
        py_ops_not_defined = NoOperatorsDefined(3)
        jit_ops_defined = JitOperatorsDefined(2)
        jit_ops_not_defined = JitNoOperatorsDefined(3)
        self.assertEqual(py_ops_not_defined == py_ops_defined, jit_ops_not_defined == jit_ops_defined)
        self.assertEqual(py_ops_not_defined <= py_ops_defined, jit_ops_not_defined <= jit_ops_defined)
        self.assertEqual(py_ops_not_defined < py_ops_defined, jit_ops_not_defined < jit_ops_defined)
        self.assertEqual(py_ops_not_defined >= py_ops_defined, jit_ops_not_defined >= jit_ops_defined)
        self.assertEqual(py_ops_not_defined > py_ops_defined, jit_ops_not_defined > jit_ops_defined)

    @skip_unless_scipy
    def test_matmul_operator(self):

        class ArrayAt:

            def __init__(self, array):
                self.arr = array

            def __matmul__(self, other):
                return self.arr @ other.arr

            def __rmatmul__(self, other):
                return other.arr @ self.arr

            def __imatmul__(self, other):
                self.arr = self.arr @ other.arr
                return self

        class ArrayNoAt:

            def __init__(self, array):
                self.arr = array
        n = 3
        np.random.seed(1)
        vec = np.random.random(size=(n,))
        mat = np.random.random(size=(n, n))
        vector_noat = ArrayNoAt(vec)
        vector_at = ArrayAt(vec)
        jit_vector_noat = jitclass(ArrayNoAt, spec={'arr': float64[::1]})(vec)
        jit_vector_at = jitclass(ArrayAt, spec={'arr': float64[::1]})(vec)
        matrix_noat = ArrayNoAt(mat)
        matrix_at = ArrayAt(mat)
        jit_matrix_noat = jitclass(ArrayNoAt, spec={'arr': float64[:, ::1]})(mat)
        jit_matrix_at = jitclass(ArrayAt, spec={'arr': float64[:, ::1]})(mat)
        np.testing.assert_allclose(vector_at @ vector_noat, jit_vector_at @ jit_vector_noat)
        np.testing.assert_allclose(vector_at @ matrix_noat, jit_vector_at @ jit_matrix_noat)
        np.testing.assert_allclose(matrix_at @ vector_noat, jit_matrix_at @ jit_vector_noat)
        np.testing.assert_allclose(matrix_at @ matrix_noat, jit_matrix_at @ jit_matrix_noat)
        np.testing.assert_allclose(vector_noat @ vector_at, jit_vector_noat @ jit_vector_at)
        np.testing.assert_allclose(vector_noat @ matrix_at, jit_vector_noat @ jit_matrix_at)
        np.testing.assert_allclose(matrix_noat @ vector_at, jit_matrix_noat @ jit_vector_at)
        np.testing.assert_allclose(matrix_noat @ matrix_at, jit_matrix_noat @ jit_matrix_at)
        vector_at @= matrix_noat
        matrix_at @= matrix_noat
        jit_vector_at @= jit_matrix_noat
        jit_matrix_at @= jit_matrix_noat
        np.testing.assert_allclose(vector_at.arr, jit_vector_at.arr)
        np.testing.assert_allclose(matrix_at.arr, jit_matrix_at.arr)

    def test_arithmetic_logical_reflection(self):

        class OperatorsDefined:

            def __init__(self, x):
                self.x = x

            def __radd__(self, other):
                return other.x + self.x

            def __rsub__(self, other):
                return other.x - self.x

            def __rmul__(self, other):
                return other.x * self.x

            def __rtruediv__(self, other):
                return other.x / self.x

            def __rfloordiv__(self, other):
                return other.x // self.x

            def __rmod__(self, other):
                return other.x % self.x

            def __rpow__(self, other):
                return other.x ** self.x

            def __rlshift__(self, other):
                return other.x << self.x

            def __rrshift__(self, other):
                return other.x >> self.x

            def __rand__(self, other):
                return other.x & self.x

            def __rxor__(self, other):
                return other.x ^ self.x

            def __ror__(self, other):
                return other.x | self.x

        class NoOperatorsDefined:

            def __init__(self, x):
                self.x = x
        float_op = ['+', '-', '*', '**', '/', '//', '%']
        int_op = [*float_op, '<<', '>>', '&', '^', '|']
        for test_type, test_op, test_value in [(int32, int_op, (2, 4)), (float64, float_op, (2.0, 4.0)), (float64[::1], float_op, (np.array([1.0, 2.0, 4.0]), np.array([20.0, -24.0, 1.0])))]:
            spec = {'x': test_type}
            JitOperatorsDefined = jitclass(OperatorsDefined, spec)
            JitNoOperatorsDefined = jitclass(NoOperatorsDefined, spec)
            py_ops_defined = OperatorsDefined(test_value[0])
            py_ops_not_defined = NoOperatorsDefined(test_value[1])
            jit_ops_defined = JitOperatorsDefined(test_value[0])
            jit_ops_not_defined = JitNoOperatorsDefined(test_value[1])
            for op in test_op:
                if not 'array' in str(test_type):
                    self.assertEqual(eval(f'py_ops_not_defined {op} py_ops_defined'), eval(f'jit_ops_not_defined {op} jit_ops_defined'))
                else:
                    self.assertTupleEqual(tuple(eval(f'py_ops_not_defined {op} py_ops_defined')), tuple(eval(f'jit_ops_not_defined {op} jit_ops_defined')))

    def test_implicit_hash_compiles(self):

        class ImplicitHash:

            def __init__(self):
                pass

            def __eq__(self, other):
                return False
        jitted = jitclass([])(ImplicitHash)
        instance = jitted()
        self.assertFalse(instance == instance)