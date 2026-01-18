import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
class TestCMath(BaseComplexTest):
    """
    Tests for cmath module support.
    """

    def check_predicate_func(self, pyfunc):
        self.run_unary(pyfunc, [types.boolean(tp) for tp in (types.complex128, types.complex64)], self.basic_values())

    def check_unary_func(self, pyfunc, ulps=1, values=None, returns_float=False, ignore_sign_on_zero=False):
        if returns_float:

            def sig(tp):
                return tp.underlying_float(tp)
        else:

            def sig(tp):
                return tp(tp)
        self.run_unary(pyfunc, [sig(types.complex128)], values or self.more_values(), ulps=ulps, ignore_sign_on_zero=ignore_sign_on_zero)
        self.run_unary(pyfunc, [sig(types.complex64)], values or self.basic_values(), ulps=ulps, ignore_sign_on_zero=ignore_sign_on_zero)

    def test_phase(self):
        self.check_unary_func(phase_usecase, returns_float=True)

    def test_polar(self):
        self.check_unary_func(polar_as_complex_usecase)

    def test_rect(self):

        def do_test(tp, seed_values):
            values = [(z.real, z.imag) for z in seed_values if not math.isinf(z.imag) or z.real == 0]
            float_type = tp.underlying_float
            self.run_binary(rect_usecase, [tp(float_type, float_type)], values)
        do_test(types.complex128, self.more_values())
        do_test(types.complex64, self.basic_values())

    def test_isnan(self):
        self.check_predicate_func(isnan_usecase)

    def test_isinf(self):
        self.check_predicate_func(isinf_usecase)

    def test_isfinite(self):
        self.check_predicate_func(isfinite_usecase)

    def test_exp(self):
        self.check_unary_func(exp_usecase, ulps=2)

    def test_log(self):
        self.check_unary_func(log_usecase)

    def test_log_base(self):
        values = list(itertools.product(self.more_values(), self.more_values()))
        value_types = [(types.complex128, types.complex128), (types.complex64, types.complex64)]
        self.run_binary(log_base_usecase, value_types, values, ulps=3)

    def test_log10(self):
        self.check_unary_func(log10_usecase)

    def test_sqrt(self):
        self.check_unary_func(sqrt_usecase)

    def test_acos(self):
        self.check_unary_func(acos_usecase, ulps=2)

    def test_asin(self):
        self.check_unary_func(asin_usecase, ulps=2)

    def test_atan(self):
        self.check_unary_func(atan_usecase, ulps=2, values=self.non_nan_values())

    def test_cos(self):
        self.check_unary_func(cos_usecase, ulps=2)

    def test_sin(self):
        self.check_unary_func(sin_usecase, ulps=2)

    def test_tan(self):
        self.check_unary_func(tan_usecase, ulps=2, ignore_sign_on_zero=True)

    def test_acosh(self):
        self.check_unary_func(acosh_usecase)

    def test_asinh(self):
        self.check_unary_func(asinh_usecase, ulps=2)

    def test_atanh(self):
        self.check_unary_func(atanh_usecase, ulps=2, ignore_sign_on_zero=True)

    def test_cosh(self):
        self.check_unary_func(cosh_usecase, ulps=2)

    def test_sinh(self):
        self.check_unary_func(sinh_usecase, ulps=2)

    def test_tanh(self):
        self.check_unary_func(tanh_usecase, ulps=2, ignore_sign_on_zero=True)