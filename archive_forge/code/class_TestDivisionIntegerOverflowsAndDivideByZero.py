import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestDivisionIntegerOverflowsAndDivideByZero:
    result_type = namedtuple('result_type', ['nocast', 'casted'])
    helper_lambdas = {'zero': lambda dtype: 0, 'min': lambda dtype: np.iinfo(dtype).min, 'neg_min': lambda dtype: -np.iinfo(dtype).min, 'min-zero': lambda dtype: (np.iinfo(dtype).min, 0), 'neg_min-zero': lambda dtype: (-np.iinfo(dtype).min, 0)}
    overflow_results = {np.remainder: result_type(helper_lambdas['zero'], helper_lambdas['zero']), np.fmod: result_type(helper_lambdas['zero'], helper_lambdas['zero']), operator.mod: result_type(helper_lambdas['zero'], helper_lambdas['zero']), operator.floordiv: result_type(helper_lambdas['min'], helper_lambdas['neg_min']), np.floor_divide: result_type(helper_lambdas['min'], helper_lambdas['neg_min']), np.divmod: result_type(helper_lambdas['min-zero'], helper_lambdas['neg_min-zero'])}

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['Integer'])
    def test_signed_division_overflow(self, dtype):
        to_check = interesting_binop_operands(np.iinfo(dtype).min, -1, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match='overflow encountered'):
                res = op1 // op2
            assert res.dtype == op1.dtype
            assert extractor(res) == np.iinfo(op1.dtype).min
            res = op1 % op2
            assert res.dtype == op1.dtype
            assert extractor(res) == 0
            res = np.fmod(op1, op2)
            assert extractor(res) == 0
            with pytest.warns(RuntimeWarning, match='overflow encountered'):
                res1, res2 = np.divmod(op1, op2)
            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == np.iinfo(op1.dtype).min
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
    def test_divide_by_zero(self, dtype):
        to_check = interesting_binop_operands(1, 0, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match='divide by zero'):
                res = op1 // op2
            assert res.dtype == op1.dtype
            assert extractor(res) == 0
            with pytest.warns(RuntimeWarning, match='divide by zero'):
                res1, res2 = np.divmod(op1, op2)
            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == 0
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dividend_dtype', np.sctypes['int'])
    @pytest.mark.parametrize('divisor_dtype', np.sctypes['int'])
    @pytest.mark.parametrize('operation', [np.remainder, np.fmod, np.divmod, np.floor_divide, operator.mod, operator.floordiv])
    @np.errstate(divide='warn', over='warn')
    def test_overflows(self, dividend_dtype, divisor_dtype, operation):
        arrays = [np.array([np.iinfo(dividend_dtype).min] * i, dtype=dividend_dtype) for i in range(1, 129)]
        divisor = np.array([-1], dtype=divisor_dtype)
        if np.dtype(dividend_dtype).itemsize >= np.dtype(divisor_dtype).itemsize and operation in (np.divmod, np.floor_divide, operator.floordiv):
            with pytest.warns(RuntimeWarning, match='overflow encountered in'):
                result = operation(dividend_dtype(np.iinfo(dividend_dtype).min), divisor_dtype(-1))
                assert result == self.overflow_results[operation].nocast(dividend_dtype)
            for a in arrays:
                with pytest.warns(RuntimeWarning, match='overflow encountered in'):
                    result = np.array(operation(a, divisor)).flatten('f')
                    expected_array = np.array([self.overflow_results[operation].nocast(dividend_dtype)] * len(a)).flatten()
                    assert_array_equal(result, expected_array)
        else:
            result = operation(dividend_dtype(np.iinfo(dividend_dtype).min), divisor_dtype(-1))
            assert result == self.overflow_results[operation].casted(dividend_dtype)
            for a in arrays:
                result = np.array(operation(a, divisor)).flatten('f')
                expected_array = np.array([self.overflow_results[operation].casted(dividend_dtype)] * len(a)).flatten()
                assert_array_equal(result, expected_array)