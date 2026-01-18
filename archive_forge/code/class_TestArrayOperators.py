import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class TestArrayOperators(BaseUFuncTest, TestCase):

    def _check_results(self, expected, got):
        self.assertEqual(expected.dtype.kind, got.dtype.kind)
        np.testing.assert_array_almost_equal(expected, got)

    def unary_op_test(self, operator, nrt=True, skip_inputs=[], additional_inputs=[], int_output_type=None, float_output_type=None):
        operator_func = _make_unary_ufunc_op_usecase(operator)
        inputs = list(self.inputs)
        inputs.extend(additional_inputs)
        pyfunc = operator_func
        for input_tuple in inputs:
            input_operand, input_type = input_tuple
            if input_type in skip_inputs or not isinstance(input_type, types.Array):
                continue
            cfunc = self._compile(pyfunc, (input_type,), nrt=nrt)
            expected = pyfunc(input_operand)
            got = cfunc(input_operand)
            self._check_results(expected, got)

    def binary_op_test(self, operator, nrt=True, skip_inputs=[], additional_inputs=[], int_output_type=None, float_output_type=None, positive_rhs=False):
        operator_func = _make_binary_ufunc_op_usecase(operator)
        inputs = list(self.inputs)
        inputs.extend(additional_inputs)
        pyfunc = operator_func
        random_state = np.random.RandomState(1)
        for input_tuple in inputs:
            input_operand1, input_type = input_tuple
            input_dtype = numpy_support.as_dtype(getattr(input_type, 'dtype', input_type))
            input_type1 = input_type
            if input_type in skip_inputs:
                continue
            if positive_rhs:
                zero = np.zeros(1, dtype=input_dtype)[0]
            if isinstance(input_type, types.Array):
                input_operand0 = input_operand1
                input_type0 = input_type
                if positive_rhs and np.any(input_operand1 < zero):
                    continue
            else:
                input_operand0 = random_state.uniform(0, 100, 10).astype(input_dtype)
                input_type0 = typeof(input_operand0)
                if positive_rhs and input_operand1 < zero:
                    continue
            args = (input_type0, input_type1)
            cfunc = self._compile(pyfunc, args, nrt=nrt)
            expected = pyfunc(input_operand0, input_operand1)
            got = cfunc(input_operand0, input_operand1)
            self._check_results(expected, got)

    def bitwise_additional_inputs(self):
        return [(True, types.boolean), (False, types.boolean), (np.array([True, False]), types.Array(types.boolean, 1, 'C'))]

    def binary_int_op_test(self, *args, **kws):
        skip_inputs = kws.setdefault('skip_inputs', [])
        skip_inputs += [types.float32, types.float64, types.Array(types.float32, 1, 'C'), types.Array(types.float64, 1, 'C')]
        return self.binary_op_test(*args, **kws)

    def binary_bitwise_op_test(self, *args, **kws):
        additional_inputs = kws.setdefault('additional_inputs', [])
        additional_inputs += self.bitwise_additional_inputs()
        return self.binary_int_op_test(*args, **kws)

    def inplace_op_test(self, operator, lhs_values, rhs_values, lhs_dtypes, rhs_dtypes, precise=True):
        operator_func = _make_inplace_ufunc_op_usecase(operator)
        pyfunc = operator_func
        if precise:
            assertion = self.assertPreciseEqual
        else:
            assertion = np.testing.assert_allclose
        lhs_inputs = [np.array(lhs_values, dtype=dtype) for dtype in lhs_dtypes]
        rhs_arrays = [np.array(rhs_values, dtype=dtype) for dtype in rhs_dtypes]
        rhs_scalars = [dtype(v) for v in rhs_values for dtype in rhs_dtypes]
        rhs_inputs = rhs_arrays + rhs_scalars
        for lhs, rhs in itertools.product(lhs_inputs, rhs_inputs):
            lhs_type = typeof(lhs)
            rhs_type = typeof(rhs)
            args = (lhs_type, rhs_type)
            cfunc = self._compile(pyfunc, args)
            expected = lhs.copy()
            pyfunc(expected, rhs)
            got = lhs.copy()
            cfunc(got, rhs)
            assertion(got, expected)

    def inplace_float_op_test(self, operator, lhs_values, rhs_values, precise=True):
        return self.inplace_op_test(operator, lhs_values, rhs_values, (np.float32, np.float64), (np.float32, np.float64, np.int64), precise=precise)

    def inplace_int_op_test(self, operator, lhs_values, rhs_values):
        self.inplace_op_test(operator, lhs_values, rhs_values, (np.int16, np.int32, np.int64), (np.int16, np.uint32))

    def inplace_bitwise_op_test(self, operator, lhs_values, rhs_values):
        self.inplace_int_op_test(operator, lhs_values, rhs_values)
        self.inplace_op_test(operator, lhs_values, rhs_values, (np.bool_,), (np.bool_, np.bool_))

    def test_unary_positive_array_op(self):
        self.unary_op_test('+')

    def test_unary_negative_array_op(self):
        self.unary_op_test('-')

    def test_unary_invert_array_op(self):
        self.unary_op_test('~', skip_inputs=[types.float32, types.float64, types.Array(types.float32, 1, 'C'), types.Array(types.float64, 1, 'C')], additional_inputs=self.bitwise_additional_inputs())

    def test_inplace_add(self):
        self.inplace_float_op_test('+=', [-1, 1.5, 3], [-5, 0, 2.5])
        self.inplace_float_op_test(operator.iadd, [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_sub(self):
        self.inplace_float_op_test('-=', [-1, 1.5, 3], [-5, 0, 2.5])
        self.inplace_float_op_test(operator.isub, [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_mul(self):
        self.inplace_float_op_test('*=', [-1, 1.5, 3], [-5, 0, 2.5])
        self.inplace_float_op_test(operator.imul, [-1, 1.5, 3], [-5, 0, 2.5])

    def test_inplace_floordiv(self):
        self.inplace_float_op_test('//=', [-1, 1.5, 3], [-5, 1.25, 2.5])
        self.inplace_float_op_test(operator.ifloordiv, [-1, 1.5, 3], [-5, 1.25, 2.5])

    def test_inplace_div(self):
        self.inplace_float_op_test('/=', [-1, 1.5, 3], [-5, 0, 2.5])
        self.inplace_float_op_test(operator.itruediv, [-1, 1.5, 3], [-5, 1.25, 2.5])

    def test_inplace_remainder(self):
        self.inplace_float_op_test('%=', [-1, 1.5, 3], [-5, 2, 2.5])
        self.inplace_float_op_test(operator.imod, [-1, 1.5, 3], [-5, 2, 2.5])

    def test_inplace_pow(self):
        self.inplace_float_op_test('**=', [-1, 1.5, 3], [-5, 2, 2.5], precise=False)
        self.inplace_float_op_test(operator.ipow, [-1, 1.5, 3], [-5, 2, 2.5], precise=False)

    def test_inplace_and(self):
        self.inplace_bitwise_op_test('&=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])
        self.inplace_bitwise_op_test(operator.iand, [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_or(self):
        self.inplace_bitwise_op_test('|=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])
        self.inplace_bitwise_op_test(operator.ior, [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_xor(self):
        self.inplace_bitwise_op_test('^=', [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])
        self.inplace_bitwise_op_test(operator.ixor, [0, 1, 2, 3, 51], [0, 13, 16, 42, 255])

    def test_inplace_lshift(self):
        self.inplace_int_op_test('<<=', [0, 5, -10, -51], [0, 1, 4, 14])
        self.inplace_int_op_test(operator.ilshift, [0, 5, -10, -51], [0, 1, 4, 14])

    def test_inplace_rshift(self):
        self.inplace_int_op_test('>>=', [0, 5, -10, -51], [0, 1, 4, 14])
        self.inplace_int_op_test(operator.irshift, [0, 5, -10, -51], [0, 1, 4, 14])

    def test_unary_positive_array_op_2(self):
        """
        Verify that the unary positive operator copies values, and doesn't
        just alias to the input array (mirrors normal Numpy/Python
        interaction behavior).
        """

        def f(a1):
            a2 = +a1
            a1[0] = 3
            a2[1] = 4
            return a2
        a1 = np.zeros(10)
        a2 = f(a1)
        self.assertTrue(a1[0] != a2[0] and a1[1] != a2[1])
        a3 = np.zeros(10)
        a4 = njit(f)(a3)
        self.assertTrue(a3[0] != a4[0] and a3[1] != a4[1])
        np.testing.assert_array_equal(a1, a3)
        np.testing.assert_array_equal(a2, a4)

    def test_add_array_op(self):
        self.binary_op_test('+')

    def test_subtract_array_op(self):
        self.binary_op_test('-')

    def test_multiply_array_op(self):
        self.binary_op_test('*')

    def test_divide_array_op(self):
        int_out_type = None
        int_out_type = types.float64
        self.binary_op_test('/', int_output_type=int_out_type)

    def test_floor_divide_array_op(self):
        self.inputs = [(np.uint32(1), types.uint32), (np.int32(-2), types.int32), (np.int32(0), types.int32), (np.uint64(4), types.uint64), (np.int64(-5), types.int64), (np.int64(0), types.int64), (np.float32(-0.5), types.float32), (np.float32(1.5), types.float32), (np.float64(-2.5), types.float64), (np.float64(3.5), types.float64), (np.array([1, 2], dtype='u4'), types.Array(types.uint32, 1, 'C')), (np.array([3, 4], dtype='u8'), types.Array(types.uint64, 1, 'C')), (np.array([-1, 1, 5], dtype='i4'), types.Array(types.int32, 1, 'C')), (np.array([-1, 1, 6], dtype='i8'), types.Array(types.int64, 1, 'C')), (np.array([-0.5, 1.5], dtype='f4'), types.Array(types.float32, 1, 'C')), (np.array([-2.5, 3.5], dtype='f8'), types.Array(types.float64, 1, 'C'))]
        self.binary_op_test('//')

    def test_remainder_array_op(self):
        self.binary_op_test('%')

    def test_power_array_op(self):
        self.binary_op_test('**', positive_rhs=True)

    def test_left_shift_array_op(self):
        self.binary_int_op_test('<<', positive_rhs=True)

    def test_right_shift_array_op(self):
        self.binary_int_op_test('>>', positive_rhs=True)

    def test_bitwise_and_array_op(self):
        self.binary_bitwise_op_test('&')

    def test_bitwise_or_array_op(self):
        self.binary_bitwise_op_test('|')

    def test_bitwise_xor_array_op(self):
        self.binary_bitwise_op_test('^')

    def test_equal_array_op(self):
        self.binary_op_test('==')

    def test_greater_array_op(self):
        self.binary_op_test('>')

    def test_greater_equal_array_op(self):
        self.binary_op_test('>=')

    def test_less_array_op(self):
        self.binary_op_test('<')

    def test_less_equal_array_op(self):
        self.binary_op_test('<=')

    def test_not_equal_array_op(self):
        self.binary_op_test('!=')