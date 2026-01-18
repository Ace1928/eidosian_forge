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
class TestBuiltins(TestCase):

    def run_nullary_func(self, pyfunc, flags):
        cfunc = jit((), **flags)(pyfunc)
        expected = pyfunc()
        self.assertPreciseEqual(cfunc(), expected)

    def test_abs(self, flags=forceobj_flags):
        pyfunc = abs_usecase
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cfunc = jit((types.float32,), **flags)(pyfunc)
        for x in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')
        complex_values = [-1.1 + 0.5j, 0.0 + 0j, 1.1 + 3j, float('inf') + 1j * float('nan'), float('nan') - 1j * float('inf')]
        cfunc = jit((types.complex64,), **flags)(pyfunc)
        for x in complex_values:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')
        cfunc = jit((types.complex128,), **flags)(pyfunc)
        for x in complex_values:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        for unsigned_type in types.unsigned_domain:
            unsigned_values = [0, 10, 2, 2 ** unsigned_type.bitwidth - 1]
            cfunc = jit((unsigned_type,), **flags)(pyfunc)
            for x in unsigned_values:
                self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_abs_npm(self):
        self.test_abs(flags=no_pyobj_flags)

    def test_all(self, flags=forceobj_flags):
        pyfunc = all_usecase
        cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_all_npm(self):
        with self.assertTypingError():
            self.test_all(flags=no_pyobj_flags)

    def test_any(self, flags=forceobj_flags):
        pyfunc = any_usecase
        cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
        x_operands = [-1, 0, 1, None]
        y_operands = [-1, 0, 1, None]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_any_npm(self):
        with self.assertTypingError():
            self.test_any(flags=no_pyobj_flags)

    def test_bool(self, flags=forceobj_flags):
        pyfunc = bool_usecase
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cfunc = jit((types.float64,), **flags)(pyfunc)
        for x in [0.0, -0.0, 1.5, float('inf'), float('nan')]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cfunc = jit((types.complex128,), **flags)(pyfunc)
        for x in [complex(0, float('inf')), complex(0, float('nan'))]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_bool_npm(self):
        self.test_bool(flags=no_pyobj_flags)

    def test_bool_nonnumber(self, flags=forceobj_flags):
        pyfunc = bool_usecase
        cfunc = jit((types.string,), **flags)(pyfunc)
        for x in ['x', '']:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cfunc = jit((types.Dummy('list'),), **flags)(pyfunc)
        for x in [[1], []]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_bool_nonnumber_npm(self):
        with self.assertTypingError():
            self.test_bool_nonnumber(flags=no_pyobj_flags)

    def test_complex(self, flags=forceobj_flags):
        pyfunc = complex_usecase
        cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_complex_npm(self):
        self.test_complex(flags=no_pyobj_flags)

    def test_divmod_ints(self, flags=forceobj_flags):
        pyfunc = divmod_usecase
        cfunc = jit((types.int64, types.int64), **flags)(pyfunc)

        def truncate_result(x, bits=64):
            if x >= 0:
                x &= (1 << bits - 1) - 1
            return x
        denominators = [1, 3, 7, 15, -1, -3, -7, -15, 2 ** 63 - 1, -2 ** 63]
        numerators = denominators + [0]
        for x, y in itertools.product(numerators, denominators):
            expected_quot, expected_rem = pyfunc(x, y)
            quot, rem = cfunc(x, y)
            f = truncate_result
            self.assertPreciseEqual((f(quot), f(rem)), (f(expected_quot), f(expected_rem)))
        for x in numerators:
            with self.assertRaises(ZeroDivisionError):
                cfunc(x, 0)

    def test_divmod_ints_npm(self):
        self.test_divmod_ints(flags=no_pyobj_flags)

    def test_divmod_floats(self, flags=forceobj_flags):
        pyfunc = divmod_usecase
        cfunc = jit((types.float64, types.float64), **flags)(pyfunc)
        denominators = [1.0, 3.5, 1e+100, -2.0, -7.5, -1e+101, np.inf, -np.inf, np.nan]
        numerators = denominators + [-0.0, 0.0]
        for x, y in itertools.product(numerators, denominators):
            expected_quot, expected_rem = pyfunc(x, y)
            quot, rem = cfunc(x, y)
            self.assertPreciseEqual((quot, rem), (expected_quot, expected_rem))
        for x in numerators:
            with self.assertRaises(ZeroDivisionError):
                cfunc(x, 0.0)

    def test_divmod_floats_npm(self):
        self.test_divmod_floats(flags=no_pyobj_flags)

    def test_enumerate(self, flags=forceobj_flags):
        self.run_nullary_func(enumerate_usecase, flags)

    def test_enumerate_npm(self):
        self.test_enumerate(flags=no_pyobj_flags)

    def test_enumerate_start(self, flags=forceobj_flags):
        self.run_nullary_func(enumerate_start_usecase, flags)

    def test_enumerate_start_npm(self):
        self.test_enumerate_start(flags=no_pyobj_flags)

    def test_enumerate_start_invalid_start_type(self):
        pyfunc = enumerate_invalid_start_usecase
        cfunc = jit((), **forceobj_flags)(pyfunc)
        with self.assertRaises(TypeError) as raises:
            cfunc()
        msg = "'float' object cannot be interpreted as an integer"
        self.assertIn(msg, str(raises.exception))

    def test_enumerate_start_invalid_start_type_npm(self):
        pyfunc = enumerate_invalid_start_usecase
        with self.assertRaises(errors.TypingError) as raises:
            jit((), **no_pyobj_flags)(pyfunc)
        msg = 'Only integers supported as start value in enumerate'
        self.assertIn(msg, str(raises.exception))

    def test_filter(self, flags=forceobj_flags):
        pyfunc = filter_usecase
        argtys = (types.Dummy('list'), types.Dummy('function_ptr'))
        cfunc = jit(argtys, **flags)(pyfunc)
        filter_func = lambda x: x % 2
        x = [0, 1, 2, 3, 4]
        self.assertSequenceEqual(list(cfunc(x, filter_func)), list(pyfunc(x, filter_func)))

    def test_filter_npm(self):
        with self.assertTypingError():
            self.test_filter(flags=no_pyobj_flags)

    def test_float(self, flags=forceobj_flags):
        pyfunc = float_usecase
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))
        cfunc = jit((types.float32,), **flags)(pyfunc)
        for x in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x), prec='single')
        cfunc = jit((types.string,), **flags)(pyfunc)
        for x in ['-1.1', '0.0', '1.1']:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_float_npm(self):
        with self.assertTypingError():
            self.test_float(flags=no_pyobj_flags)

    def test_format(self, flags=forceobj_flags):
        pyfunc = format_usecase
        cfunc = jit((types.string, types.int32), **flags)(pyfunc)
        x = '{0}'
        for y in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))
        cfunc = jit((types.string, types.float32), **flags)(pyfunc)
        x = '{0}'
        for y in [-1.1, 0.0, 1.1]:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))
        cfunc = jit((types.string, types.string), **flags)(pyfunc)
        x = '{0}'
        for y in ['a', 'b', 'c']:
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_format_npm(self):
        with self.assertTypingError():
            self.test_format(flags=no_pyobj_flags)

    def test_globals(self, flags=forceobj_flags):
        pyfunc = globals_usecase
        cfunc = jit((), **flags)(pyfunc)
        g = cfunc()
        self.assertIs(g, globals())

    def test_globals_npm(self):
        with self.assertTypingError():
            self.test_globals(flags=no_pyobj_flags)

    def test_globals_jit(self, flags=forceobj_flags):
        pyfunc = globals_usecase
        jitted = jit(**flags)(pyfunc)
        self.assertIs(jitted(), globals())
        self.assertIs(jitted(), globals())

    def test_globals_jit_npm(self):
        with self.assertTypingError():
            self.test_globals_jit(nopython=True)

    def test_hex(self, flags=forceobj_flags):
        pyfunc = hex_usecase
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_hex_npm(self):
        with self.assertTypingError():
            self.test_hex(flags=no_pyobj_flags)

    def test_int_str(self):
        pyfunc = str_usecase
        small_inputs = [1234, 1, 0, 10, 1000]
        large_inputs = [123456789, 2222222, 1000000, ~0]
        args = [*small_inputs, *large_inputs]
        typs = [types.int8, types.int16, types.int32, types.int64, types.uint, types.uint8, types.uint16, types.uint32, types.uint64]
        for typ in typs:
            cfunc = jit((typ,), **nrt_no_pyobj_flags)(pyfunc)
            for v in args:
                self.assertPreciseEqual(cfunc(typ(v)), pyfunc(typ(v)))
                if typ.signed:
                    self.assertPreciseEqual(cfunc(typ(-v)), pyfunc(typ(-v)))

    def test_int(self, flags=forceobj_flags):
        pyfunc = int_usecase
        cfunc = jit((types.string, types.int32), **flags)(pyfunc)
        x_operands = ['-1', '0', '1', '10']
        y_operands = [2, 8, 10, 16]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_int_npm(self):
        with self.assertTypingError():
            self.test_int(flags=no_pyobj_flags)

    def test_iter_next(self, flags=forceobj_flags):
        pyfunc = iter_next_usecase
        cfunc = jit((types.UniTuple(types.int32, 3),), **flags)(pyfunc)
        self.assertPreciseEqual(cfunc((1, 42, 5)), (1, 42))
        cfunc = jit((types.UniTuple(types.int32, 1),), **flags)(pyfunc)
        with self.assertRaises(StopIteration):
            cfunc((1,))

    def test_iter_next_npm(self):
        self.test_iter_next(flags=no_pyobj_flags)

    def test_locals(self, flags=forceobj_flags):
        pyfunc = locals_usecase
        with self.assertRaises(errors.ForbiddenConstruct):
            jit((types.int64,), **flags)(pyfunc)

    def test_locals_forceobj(self):
        self.test_locals(flags=forceobj_flags)

    def test_locals_npm(self):
        with self.assertTypingError():
            self.test_locals(flags=no_pyobj_flags)

    def test_map(self, flags=forceobj_flags):
        pyfunc = map_usecase
        argtys = (types.Dummy('list'), types.Dummy('function_ptr'))
        cfunc = jit(argtys, **flags)(pyfunc)
        map_func = lambda x: x * 2
        x = [0, 1, 2, 3, 4]
        self.assertSequenceEqual(list(cfunc(x, map_func)), list(pyfunc(x, map_func)))

    def test_map_npm(self):
        with self.assertTypingError():
            self.test_map(flags=no_pyobj_flags)

    def check_minmax_1(self, pyfunc, flags):
        cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_1(self, flags=forceobj_flags):
        """
        max(*args)
        """
        self.check_minmax_1(max_usecase1, flags)

    def test_min_1(self, flags=forceobj_flags):
        """
        min(*args)
        """
        self.check_minmax_1(min_usecase1, flags)

    def test_max_npm_1(self):
        self.test_max_1(flags=no_pyobj_flags)

    def test_min_npm_1(self):
        self.test_min_1(flags=no_pyobj_flags)

    def check_minmax_2(self, pyfunc, flags):
        cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
        x_operands = [-1, 0, 1]
        y_operands = [-1, 0, 1]
        for x, y in itertools.product(x_operands, y_operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_2(self, flags=forceobj_flags):
        """
        max(list)
        """
        self.check_minmax_2(max_usecase2, flags)

    def test_min_2(self, flags=forceobj_flags):
        """
        min(list)
        """
        self.check_minmax_2(min_usecase2, flags)

    def test_max_npm_2(self):
        with self.assertTypingError():
            self.test_max_2(flags=no_pyobj_flags)

    def test_min_npm_2(self):
        with self.assertTypingError():
            self.test_min_2(flags=no_pyobj_flags)

    def check_minmax_3(self, pyfunc, flags):

        def check(argty):
            cfunc = jit((argty,), **flags)(pyfunc)
            tup = (1.5, float('nan'), 2.5)
            for val in [tup, tup[::-1]]:
                self.assertPreciseEqual(cfunc(val), pyfunc(val))
        check(types.UniTuple(types.float64, 3))
        check(types.Tuple((types.float32, types.float64, types.float32)))

    def test_max_3(self, flags=forceobj_flags):
        """
        max(tuple)
        """
        self.check_minmax_3(max_usecase3, flags)

    def test_min_3(self, flags=forceobj_flags):
        """
        min(tuple)
        """
        self.check_minmax_3(min_usecase3, flags)

    def test_max_npm_3(self):
        self.test_max_3(flags=no_pyobj_flags)

    def test_min_npm_3(self):
        self.test_min_3(flags=no_pyobj_flags)

    def check_min_max_invalid_types(self, pyfunc, flags=forceobj_flags):
        cfunc = jit((types.int32, types.Dummy('list')), **flags)(pyfunc)
        cfunc(1, [1])

    def test_max_1_invalid_types(self):
        with self.assertRaises(TypeError):
            self.check_min_max_invalid_types(max_usecase1)

    def test_max_1_invalid_types_npm(self):
        with self.assertTypingError():
            self.check_min_max_invalid_types(max_usecase1, flags=no_pyobj_flags)

    def test_min_1_invalid_types(self):
        with self.assertRaises(TypeError):
            self.check_min_max_invalid_types(min_usecase1)

    def test_min_1_invalid_types_npm(self):
        with self.assertTypingError():
            self.check_min_max_invalid_types(min_usecase1, flags=no_pyobj_flags)

    def check_minmax_bool1(self, pyfunc, flags):
        cfunc = jit((types.bool_, types.bool_), **flags)(pyfunc)
        operands = (False, True)
        for x, y in itertools.product(operands, operands):
            self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))

    def test_max_bool1(self, flags=forceobj_flags):
        self.check_minmax_bool1(max_usecase1, flags)

    def test_min_bool1(self, flags=forceobj_flags):
        self.check_minmax_bool1(min_usecase1, flags)

    def check_min_max_unary_non_iterable(self, pyfunc, flags=forceobj_flags):
        cfunc = jit((types.int32,), **flags)(pyfunc)
        cfunc(1)

    def test_max_unary_non_iterable(self):
        with self.assertRaises(TypeError):
            self.check_min_max_unary_non_iterable(max_usecase3)

    def test_max_unary_non_iterable_npm(self):
        with self.assertTypingError():
            self.check_min_max_unary_non_iterable(max_usecase3)

    def test_min_unary_non_iterable(self):
        with self.assertRaises(TypeError):
            self.check_min_max_unary_non_iterable(min_usecase3)

    def test_min_unary_non_iterable_npm(self):
        with self.assertTypingError():
            self.check_min_max_unary_non_iterable(min_usecase3)

    def check_min_max_empty_tuple(self, pyfunc, func_name):
        with self.assertTypingError() as raises:
            jit((), **no_pyobj_flags)(pyfunc)
        self.assertIn('%s() argument is an empty tuple' % func_name, str(raises.exception))

    def test_max_empty_tuple(self):
        self.check_min_max_empty_tuple(max_usecase4, 'max')

    def test_min_empty_tuple(self):
        self.check_min_max_empty_tuple(min_usecase4, 'min')

    def test_oct(self, flags=forceobj_flags):
        pyfunc = oct_usecase
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-8, -1, 0, 1, 8]:
            self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_oct_npm(self):
        with self.assertTypingError():
            self.test_oct(flags=no_pyobj_flags)

    def test_reduce(self, flags=forceobj_flags):
        pyfunc = reduce_usecase
        argtys = (types.Dummy('function_ptr'), types.Dummy('list'))
        cfunc = jit(argtys, **flags)(pyfunc)
        reduce_func = lambda x, y: x + y
        x = range(10)
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))
        x = [x + x / 10.0 for x in range(10)]
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))
        x = [complex(x, x) for x in range(10)]
        self.assertPreciseEqual(cfunc(reduce_func, x), pyfunc(reduce_func, x))

    def test_reduce_npm(self):
        with self.assertTypingError():
            self.test_reduce(flags=no_pyobj_flags)

    def test_round1(self, flags=forceobj_flags):
        pyfunc = round_usecase1
        for tp in (types.float64, types.float32):
            cfunc = jit((tp,), **flags)(pyfunc)
            values = [-1.6, -1.5, -1.4, -0.5, 0.0, 0.1, 0.5, 0.6, 1.4, 1.5, 5.0]
            values += [-0.1, -0.0]
            for x in values:
                self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_round1_npm(self):
        self.test_round1(flags=no_pyobj_flags)

    def test_round2(self, flags=forceobj_flags):
        pyfunc = round_usecase2
        for tp in (types.float64, types.float32):
            prec = 'single' if tp is types.float32 else 'exact'
            cfunc = jit((tp, types.int32), **flags)(pyfunc)
            for x in [0.0, 0.1, 0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 12.5, 15.0, 22.5]:
                for n in (-1, 0, 1, 2):
                    self.assertPreciseEqual(cfunc(x, n), pyfunc(x, n), prec=prec)
                    expected = pyfunc(-x, n)
                    self.assertPreciseEqual(cfunc(-x, n), pyfunc(-x, n), prec=prec)

    def test_round2_npm(self):
        self.test_round2(flags=no_pyobj_flags)

    def test_sum_objmode(self, flags=forceobj_flags):
        pyfunc = sum_usecase
        cfunc = jit((types.Dummy('list'),), **flags)(pyfunc)
        x = range(10)
        self.assertPreciseEqual(cfunc(x), pyfunc(x))
        x = [x + x / 10.0 for x in range(10)]
        self.assertPreciseEqual(cfunc(x), pyfunc(x))
        x = [complex(x, x) for x in range(10)]
        self.assertPreciseEqual(cfunc(x), pyfunc(x))

    def test_sum(self):
        sum_default = njit(sum_usecase)
        sum_kwarg = njit(sum_kwarg_usecase)

        @njit
        def sum_range(sz, start=0):
            tmp = range(sz)
            ret = sum(tmp, start)
            return (sum(tmp, start=start), ret)
        ntpl = namedtuple('ntpl', ['a', 'b'])

        def args():
            yield [*range(10)]
            yield [x + x / 10.0 for x in range(10)]
            yield [x * 1j for x in range(10)]
            yield (1, 2, 3)
            yield (1, 2, 3j)
            yield (np.int64(32), np.int32(2), np.int8(3))
            tl = typed.List(range(5))
            yield tl
            yield np.ones(5)
            yield ntpl(100, 200)
            yield ntpl(100, 200j)
        for x in args():
            self.assertPreciseEqual(sum_default(x), sum_default.py_func(x))
        x = (np.uint64(32), np.uint32(2), np.uint8(3))
        self.assertEqual(sum_default(x), sum_default.py_func(x))

        def args_kws():
            yield ([*range(10)], 12)
            yield ([x + x / 10.0 for x in range(10)], 19j)
            yield ([x * 1j for x in range(10)], -2)
            yield ((1, 2, 3), 9)
            yield ((1, 2, 3j), -0)
            yield ((np.int64(32), np.int32(2), np.int8(3)), np.uint32(7))
            tl = typed.List(range(5))
            yield (tl, 100)
            yield (np.ones((5, 5)), 10 * np.ones((5,)))
            yield (ntpl(100, 200), -50)
            yield (ntpl(100, 200j), 9)
        for x, start in args_kws():
            self.assertPreciseEqual(sum_kwarg(x, start=start), sum_kwarg.py_func(x, start=start))
        for start in range(-3, 4):
            for sz in range(-3, 4):
                self.assertPreciseEqual(sum_range(sz, start=start), sum_range.py_func(sz, start=start))

    def test_sum_exceptions(self):
        sum_default = njit(sum_usecase)
        sum_kwarg = njit(sum_kwarg_usecase)
        msg = "sum() can't sum {}"
        with self.assertRaises(errors.TypingError) as raises:
            sum_kwarg((1, 2, 3), 'a')
        self.assertIn(msg.format('strings'), str(raises.exception))
        with self.assertRaises(errors.TypingError) as raises:
            sum_kwarg((1, 2, 3), b'123')
        self.assertIn(msg.format('bytes'), str(raises.exception))
        with self.assertRaises(errors.TypingError) as raises:
            sum_kwarg((1, 2, 3), bytearray(b'123'))
        self.assertIn(msg.format('bytearray'), str(raises.exception))
        with self.assertRaises(errors.TypingError) as raises:
            sum_default('abcd')
        self.assertIn('No implementation', str(raises.exception))

    def test_truth(self):
        pyfunc = truth_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(pyfunc(True), cfunc(True))
        self.assertEqual(pyfunc(False), cfunc(False))

    def test_type_unary(self):
        pyfunc = type_unary_usecase
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args):
            expected = pyfunc(*args)
            self.assertPreciseEqual(cfunc(*args), expected)
        check(1.5, 2)
        check(1, 2.5)
        check(1.5j, 2)
        check(True, 2)
        check(2.5j, False)

    def test_zip(self, flags=forceobj_flags):
        self.run_nullary_func(zip_usecase, flags)

    def test_zip_npm(self):
        self.test_zip(flags=no_pyobj_flags)

    def test_zip_1(self, flags=forceobj_flags):
        self.run_nullary_func(zip_1_usecase, flags)

    def test_zip_1_npm(self):
        self.test_zip_1(flags=no_pyobj_flags)

    def test_zip_3(self, flags=forceobj_flags):
        self.run_nullary_func(zip_3_usecase, flags)

    def test_zip_3_npm(self):
        self.test_zip_3(flags=no_pyobj_flags)

    def test_zip_0(self, flags=forceobj_flags):
        self.run_nullary_func(zip_0_usecase, flags)

    def test_zip_0_npm(self):
        self.test_zip_0(flags=no_pyobj_flags)

    def test_zip_first_exhausted(self, flags=forceobj_flags):
        """
        Test side effect to the input iterators when a left iterator has been
        exhausted before the ones on the right.
        """
        self.run_nullary_func(zip_first_exhausted, flags)

    def test_zip_first_exhausted_npm(self):
        self.test_zip_first_exhausted(flags=nrt_no_pyobj_flags)

    def test_pow_op_usecase(self):
        args = [(2, 3), (2.0, 3), (2, 3.0), (2j, 3j)]
        for x, y in args:
            argtys = (typeof(x), typeof(y))
            cfunc = jit(argtys, **no_pyobj_flags)(pow_op_usecase)
            r = cfunc(x, y)
            self.assertPreciseEqual(r, pow_op_usecase(x, y))

    def test_pow_usecase(self):
        args = [(2, 3), (2.0, 3), (2, 3.0), (2j, 3j)]
        for x, y in args:
            argtys = (typeof(x), typeof(y))
            cfunc = jit(argtys, **no_pyobj_flags)(pow_usecase)
            r = cfunc(x, y)
            self.assertPreciseEqual(r, pow_usecase(x, y))

    def _check_min_max(self, pyfunc):
        cfunc = njit()(pyfunc)
        expected = pyfunc()
        got = cfunc()
        self.assertPreciseEqual(expected, got)

    def test_min_max_iterable_input(self):

        @njit
        def frange(start, stop, step):
            i = start
            while i < stop:
                yield i
                i += step

        def sample_functions(op):
            yield (lambda: op(range(10)))
            yield (lambda: op(range(4, 12)))
            yield (lambda: op(range(-4, -15, -1)))
            yield (lambda: op([6.6, 5.5, 7.7]))
            yield (lambda: op([(3, 4), (1, 2)]))
            yield (lambda: op(frange(1.1, 3.3, 0.1)))
            yield (lambda: op([np.nan, -np.inf, np.inf, np.nan]))
            yield (lambda: op([(3,), (1,), (2,)]))
        for fn in sample_functions(op=min):
            self._check_min_max(fn)
        for fn in sample_functions(op=max):
            self._check_min_max(fn)