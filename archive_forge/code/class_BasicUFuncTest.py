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
class BasicUFuncTest(BaseUFuncTest):

    def _make_ufunc_usecase(self, ufunc):
        return _make_ufunc_usecase(ufunc)

    def basic_ufunc_test(self, ufunc, skip_inputs=[], additional_inputs=[], int_output_type=None, float_output_type=None, kinds='ifc', positive_only=False):
        self.reset_module_warnings(__name__)
        pyfunc = self._make_ufunc_usecase(ufunc)
        inputs = list(self.inputs) + additional_inputs
        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]
            is_tuple = isinstance(input_operand, tuple)
            if is_tuple:
                args = input_operand
            else:
                args = (input_operand,) * ufunc.nin
            if input_type in skip_inputs:
                continue
            if positive_only and np.any(args[0] < 0):
                continue
            if args[0].dtype.kind not in kinds:
                continue
            output_type = self._determine_output_type(input_type, int_output_type, float_output_type)
            input_types = (input_type,) * ufunc.nin
            output_types = (output_type,) * ufunc.nout
            argtys = input_types + output_types
            cfunc = self._compile(pyfunc, argtys)
            if isinstance(args[0], np.ndarray):
                results = [np.zeros(args[0].shape, dtype=out_ty.dtype.name) for out_ty in output_types]
                expected = [np.zeros(args[0].shape, dtype=out_ty.dtype.name) for out_ty in output_types]
            else:
                results = [np.zeros(1, dtype=out_ty.dtype.name) for out_ty in output_types]
                expected = [np.zeros(1, dtype=out_ty.dtype.name) for out_ty in output_types]
            invalid_flag = False
            with warnings.catch_warnings(record=True) as warnlist:
                warnings.simplefilter('always')
                pyfunc(*args, *expected)
                warnmsg = 'invalid value encountered'
                for thiswarn in warnlist:
                    if issubclass(thiswarn.category, RuntimeWarning) and str(thiswarn.message).startswith(warnmsg):
                        invalid_flag = True
            cfunc(*args, *results)
            for expected_i, result_i in zip(expected, results):
                msg = '\n'.join(["ufunc '{0}' failed", 'inputs ({1}):', '{2}', 'got({3})', '{4}', 'expected ({5}):', '{6}']).format(ufunc.__name__, input_type, input_operand, output_type, result_i, expected_i.dtype, expected_i)
                try:
                    np.testing.assert_array_almost_equal(expected_i, result_i, decimal=5, err_msg=msg)
                except AssertionError:
                    if invalid_flag:
                        print('Output mismatch for invalid input', input_tuple, result_i, expected_i)
                    else:
                        raise

    def signed_unsigned_cmp_test(self, comparison_ufunc):
        self.basic_ufunc_test(comparison_ufunc)
        if numpy_support.numpy_version < (1, 25):
            return
        additional_inputs = ((np.int64(-1), np.uint64(0)), (np.int64(-1), np.uint64(1)), (np.int64(0), np.uint64(0)), (np.int64(0), np.uint64(1)), (np.int64(1), np.uint64(0)), (np.int64(1), np.uint64(1)), (np.uint64(0), np.int64(-1)), (np.uint64(0), np.int64(0)), (np.uint64(0), np.int64(1)), (np.uint64(1), np.int64(-1)), (np.uint64(1), np.int64(0)), (np.uint64(1), np.int64(1)), (np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64), np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64)), (np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64), np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64)))
        pyfunc = self._make_ufunc_usecase(comparison_ufunc)
        for a, b in additional_inputs:
            input_types = (typeof(a), typeof(b))
            output_type = types.Array(types.bool_, 1, 'C')
            argtys = input_types + (output_type,)
            cfunc = self._compile(pyfunc, argtys)
            if isinstance(a, np.ndarray):
                result = np.zeros(a.shape, dtype=np.bool_)
            else:
                result = np.zeros(1, dtype=np.bool_)
            expected = np.zeros_like(result)
            pyfunc(a, b, expected)
            cfunc(a, b, result)
            np.testing.assert_equal(expected, result)