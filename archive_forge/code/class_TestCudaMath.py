import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
class TestCudaMath(CUDATestCase):

    def unary_template_float16(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float16, np.float16, start, stop)

    def unary_template_float32(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float32, np.float32, start, stop)

    def unary_template_float64(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float64, np.float64, start, stop)

    def unary_template_int64(self, func, npfunc, start=0, stop=50):
        self.unary_template(func, npfunc, np.int64, np.float64, start, stop)

    def unary_template_uint64(self, func, npfunc, start=0, stop=50):
        self.unary_template(func, npfunc, np.uint64, np.float64, start, stop)

    def unary_template(self, func, npfunc, npdtype, nprestype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty_like(A).astype(nprestype)
        arytype = numpy_support.from_dtype(npdtype)[::1]
        restype = numpy_support.from_dtype(nprestype)[::1]
        cfunc = cuda.jit((arytype, restype))(func)
        cfunc[1, nelem](A, B)
        if npdtype == np.float64:
            rtol = 1e-13
        elif npdtype == np.float32:
            rtol = 1e-06
        else:
            rtol = 0.001
        np.testing.assert_allclose(npfunc(A), B, rtol=rtol)

    def unary_bool_special_values(self, func, npfunc, npdtype, npmtype):
        fi = np.finfo(npdtype)
        denorm = fi.tiny / 4
        A = np.array([0.0, denorm, fi.tiny, 0.5, 1.0, fi.max, np.inf, np.nan], dtype=npdtype)
        B = np.empty_like(A, dtype=np.int32)
        cfunc = cuda.jit((npmtype[::1], int32[::1]))(func)
        cfunc[1, A.size](A, B)
        np.testing.assert_array_equal(B, npfunc(A))
        cfunc[1, A.size](-A, B)
        np.testing.assert_array_equal(B, npfunc(-A))

    def unary_bool_special_values_float32(self, func, npfunc):
        self.unary_bool_special_values(func, npfunc, np.float32, float32)

    def unary_bool_special_values_float64(self, func, npfunc):
        self.unary_bool_special_values(func, npfunc, np.float64, float64)

    def unary_bool_template_float32(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float32, np.float32, start, stop)

    def unary_bool_template_float64(self, func, npfunc, start=0, stop=1):
        self.unary_template(func, npfunc, np.float64, np.float64, start, stop)

    def unary_bool_template_int32(self, func, npfunc, start=0, stop=49):
        self.unary_template(func, npfunc, np.int32, np.int32, start, stop)

    def unary_bool_template_int64(self, func, npfunc, start=0, stop=49):
        self.unary_template(func, npfunc, np.int64, np.int64, start, stop)

    def unary_bool_template(self, func, npfunc, npdtype, npmtype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty(A.shape, dtype=np.int32)
        iarytype = npmtype[::1]
        oarytype = int32[::1]
        cfunc = cuda.jit((iarytype, oarytype))(func)
        cfunc[1, nelem](A, B)
        np.testing.assert_allclose(npfunc(A), B)

    def binary_template_float32(self, func, npfunc, start=0, stop=1):
        self.binary_template(func, npfunc, np.float32, np.float32, start, stop)

    def binary_template_float64(self, func, npfunc, start=0, stop=1):
        self.binary_template(func, npfunc, np.float64, np.float64, start, stop)

    def binary_template_int64(self, func, npfunc, start=0, stop=50):
        self.binary_template(func, npfunc, np.int64, np.float64, start, stop)

    def binary_template_uint64(self, func, npfunc, start=0, stop=50):
        self.binary_template(func, npfunc, np.uint64, np.float64, start, stop)

    def binary_template(self, func, npfunc, npdtype, nprestype, start, stop):
        nelem = 50
        A = np.linspace(start, stop, nelem).astype(npdtype)
        B = np.empty_like(A).astype(nprestype)
        arytype = numpy_support.from_dtype(npdtype)[::1]
        restype = numpy_support.from_dtype(nprestype)[::1]
        cfunc = cuda.jit((arytype, arytype, restype))(func)
        cfunc[1, nelem](A, A, B)
        np.testing.assert_allclose(npfunc(A, A), B)

    def test_math_acos(self):
        self.unary_template_float32(math_acos, np.arccos)
        self.unary_template_float64(math_acos, np.arccos)
        self.unary_template_int64(math_acos, np.arccos, start=0, stop=0)
        self.unary_template_uint64(math_acos, np.arccos, start=0, stop=0)

    def test_math_asin(self):
        self.unary_template_float32(math_asin, np.arcsin)
        self.unary_template_float64(math_asin, np.arcsin)
        self.unary_template_int64(math_asin, np.arcsin, start=0, stop=0)
        self.unary_template_uint64(math_asin, np.arcsin, start=0, stop=0)

    def test_math_atan(self):
        self.unary_template_float32(math_atan, np.arctan)
        self.unary_template_float64(math_atan, np.arctan)
        self.unary_template_int64(math_atan, np.arctan)
        self.unary_template_uint64(math_atan, np.arctan)

    def test_math_acosh(self):
        self.unary_template_float32(math_acosh, np.arccosh, start=1, stop=2)
        self.unary_template_float64(math_acosh, np.arccosh, start=1, stop=2)
        self.unary_template_int64(math_acosh, np.arccosh, start=1, stop=2)
        self.unary_template_uint64(math_acosh, np.arccosh, start=1, stop=2)

    def test_math_asinh(self):
        self.unary_template_float32(math_asinh, np.arcsinh)
        self.unary_template_float64(math_asinh, np.arcsinh)
        self.unary_template_int64(math_asinh, np.arcsinh)
        self.unary_template_uint64(math_asinh, np.arcsinh)

    def test_math_atanh(self):
        self.unary_template_float32(math_atanh, np.arctanh, start=0, stop=0.9)
        self.unary_template_float64(math_atanh, np.arctanh, start=0, stop=0.9)
        self.unary_template_int64(math_atanh, np.arctanh, start=0, stop=0.9)
        self.unary_template_uint64(math_atanh, np.arctanh, start=0, stop=0.9)

    def test_math_cos(self):
        self.unary_template_float32(math_cos, np.cos)
        self.unary_template_float64(math_cos, np.cos)
        self.unary_template_int64(math_cos, np.cos)
        self.unary_template_uint64(math_cos, np.cos)

    @skip_unless_cc_53
    def test_math_fp16(self):
        self.unary_template_float16(math_sin, np.sin)
        self.unary_template_float16(math_cos, np.cos)
        self.unary_template_float16(math_exp, np.exp)
        self.unary_template_float16(math_log, np.log, start=1)
        self.unary_template_float16(math_log2, np.log2, start=1)
        self.unary_template_float16(math_log10, np.log10, start=1)
        self.unary_template_float16(math_fabs, np.fabs, start=-1)
        self.unary_template_float16(math_sqrt, np.sqrt)
        self.unary_template_float16(math_ceil, np.ceil)
        self.unary_template_float16(math_floor, np.floor)

    @skip_on_cudasim('numpy does not support trunc for float16')
    @skip_unless_cc_53
    def test_math_fp16_trunc(self):
        self.unary_template_float16(math_trunc, np.trunc)

    def test_math_sin(self):
        self.unary_template_float32(math_sin, np.sin)
        self.unary_template_float64(math_sin, np.sin)
        self.unary_template_int64(math_sin, np.sin)
        self.unary_template_uint64(math_sin, np.sin)

    def test_math_tan(self):
        self.unary_template_float32(math_tan, np.tan)
        self.unary_template_float64(math_tan, np.tan)
        self.unary_template_int64(math_tan, np.tan)
        self.unary_template_uint64(math_tan, np.tan)

    def test_math_cosh(self):
        self.unary_template_float32(math_cosh, np.cosh)
        self.unary_template_float64(math_cosh, np.cosh)
        self.unary_template_int64(math_cosh, np.cosh)
        self.unary_template_uint64(math_cosh, np.cosh)

    def test_math_sinh(self):
        self.unary_template_float32(math_sinh, np.sinh)
        self.unary_template_float64(math_sinh, np.sinh)
        self.unary_template_int64(math_sinh, np.sinh)
        self.unary_template_uint64(math_sinh, np.sinh)

    def test_math_tanh(self):
        self.unary_template_float32(math_tanh, np.tanh)
        self.unary_template_float64(math_tanh, np.tanh)
        self.unary_template_int64(math_tanh, np.tanh)
        self.unary_template_uint64(math_tanh, np.tanh)

    def test_math_atan2(self):
        self.binary_template_float32(math_atan2, np.arctan2)
        self.binary_template_float64(math_atan2, np.arctan2)
        self.binary_template_int64(math_atan2, np.arctan2)
        self.binary_template_uint64(math_atan2, np.arctan2)

    def test_math_erf(self):

        @vectorize
        def ufunc(x):
            return math.erf(x)
        self.unary_template_float32(math_erf, ufunc)
        self.unary_template_float64(math_erf, ufunc)
        self.unary_template_int64(math_erf, ufunc)
        self.unary_template_uint64(math_erf, ufunc)

    def test_math_erfc(self):

        @vectorize
        def ufunc(x):
            return math.erfc(x)
        self.unary_template_float32(math_erfc, ufunc)
        self.unary_template_float64(math_erfc, ufunc)
        self.unary_template_int64(math_erfc, ufunc)
        self.unary_template_uint64(math_erfc, ufunc)

    def test_math_exp(self):
        self.unary_template_float32(math_exp, np.exp)
        self.unary_template_float64(math_exp, np.exp)
        self.unary_template_int64(math_exp, np.exp)
        self.unary_template_uint64(math_exp, np.exp)

    def test_math_expm1(self):
        self.unary_template_float32(math_expm1, np.expm1)
        self.unary_template_float64(math_expm1, np.expm1)
        self.unary_template_int64(math_expm1, np.expm1)
        self.unary_template_uint64(math_expm1, np.expm1)

    def test_math_fabs(self):
        self.unary_template_float32(math_fabs, np.fabs, start=-1)
        self.unary_template_float64(math_fabs, np.fabs, start=-1)
        self.unary_template_int64(math_fabs, np.fabs, start=-1)
        self.unary_template_uint64(math_fabs, np.fabs, start=-1)

    def test_math_gamma(self):

        @vectorize
        def ufunc(x):
            return math.gamma(x)
        self.unary_template_float32(math_gamma, ufunc, start=0.1)
        self.unary_template_float64(math_gamma, ufunc, start=0.1)
        self.unary_template_int64(math_gamma, ufunc, start=1)
        self.unary_template_uint64(math_gamma, ufunc, start=1)

    def test_math_lgamma(self):

        @vectorize
        def ufunc(x):
            return math.lgamma(x)
        self.unary_template_float32(math_lgamma, ufunc, start=0.1)
        self.unary_template_float64(math_lgamma, ufunc, start=0.1)
        self.unary_template_int64(math_lgamma, ufunc, start=1)
        self.unary_template_uint64(math_lgamma, ufunc, start=1)

    def test_math_log(self):
        self.unary_template_float32(math_log, np.log, start=1)
        self.unary_template_float64(math_log, np.log, start=1)
        self.unary_template_int64(math_log, np.log, start=1)
        self.unary_template_uint64(math_log, np.log, start=1)

    def test_math_log2(self):
        self.unary_template_float32(math_log2, np.log2, start=1)
        self.unary_template_float64(math_log2, np.log2, start=1)
        self.unary_template_int64(math_log2, np.log2, start=1)
        self.unary_template_uint64(math_log2, np.log2, start=1)

    def test_math_log10(self):
        self.unary_template_float32(math_log10, np.log10, start=1)
        self.unary_template_float64(math_log10, np.log10, start=1)
        self.unary_template_int64(math_log10, np.log10, start=1)
        self.unary_template_uint64(math_log10, np.log10, start=1)

    def test_math_log1p(self):
        self.unary_template_float32(math_log1p, np.log1p)
        self.unary_template_float64(math_log1p, np.log1p)
        self.unary_template_int64(math_log1p, np.log1p)
        self.unary_template_uint64(math_log1p, np.log1p)

    def test_math_remainder(self):
        self.binary_template_float32(math_remainder, np.remainder, start=1e-11)
        self.binary_template_float64(math_remainder, np.remainder, start=1e-11)
        self.binary_template_int64(math_remainder, np.remainder, start=1)
        self.binary_template_uint64(math_remainder, np.remainder, start=1)

    @skip_on_cudasim('math.remainder(0, 0) raises a ValueError on CUDASim')
    def test_math_remainder_0_0(self):

        @cuda.jit(void(float64[::1], int64, int64))
        def test_0_0(r, x, y):
            r[0] = math.remainder(x, y)
        r = np.zeros(1, np.float64)
        test_0_0[1, 1](r, 0, 0)
        self.assertTrue(np.isnan(r[0]))

    def test_math_sqrt(self):
        self.unary_template_float32(math_sqrt, np.sqrt)
        self.unary_template_float64(math_sqrt, np.sqrt)
        self.unary_template_int64(math_sqrt, np.sqrt)
        self.unary_template_uint64(math_sqrt, np.sqrt)

    def test_math_hypot(self):
        self.binary_template_float32(math_hypot, np.hypot)
        self.binary_template_float64(math_hypot, np.hypot)
        self.binary_template_int64(math_hypot, np.hypot)
        self.binary_template_uint64(math_hypot, np.hypot)

    def pow_template_int32(self, npdtype):
        nelem = 50
        A = np.linspace(0, 25, nelem).astype(npdtype)
        B = np.arange(nelem, dtype=np.int32)
        C = np.empty_like(A)
        arytype = numpy_support.from_dtype(npdtype)[::1]
        cfunc = cuda.jit((arytype, int32[::1], arytype))(math_pow)
        cfunc[1, nelem](A, B, C)
        Cref = np.empty_like(A)
        for i in range(len(A)):
            Cref[i] = math.pow(A[i], B[i])
        np.testing.assert_allclose(np.power(A, B).astype(npdtype), C, rtol=1e-06)

    def test_math_pow(self):
        self.binary_template_float32(math_pow, np.power)
        self.binary_template_float64(math_pow, np.power)
        self.pow_template_int32(np.float32)
        self.pow_template_int32(np.float64)

    def test_math_pow_binop(self):
        self.binary_template_float32(math_pow_binop, np.power)
        self.binary_template_float64(math_pow_binop, np.power)

    def test_math_ceil(self):
        self.unary_template_float32(math_ceil, np.ceil)
        self.unary_template_float64(math_ceil, np.ceil)
        self.unary_template_int64(math_ceil, np.ceil)
        self.unary_template_uint64(math_ceil, np.ceil)

    def test_math_floor(self):
        self.unary_template_float32(math_floor, np.floor)
        self.unary_template_float64(math_floor, np.floor)
        self.unary_template_int64(math_floor, np.floor)
        self.unary_template_uint64(math_floor, np.floor)

    def test_math_trunc(self):
        self.unary_template_float64(math_trunc, np.trunc)

    @skip_on_cudasim('trunc only supported on NumPy float64')
    def test_math_trunc_non_float64(self):
        self.unary_template_float32(math_trunc, np.trunc)
        self.unary_template_int64(math_trunc, np.trunc)
        self.unary_template_uint64(math_trunc, np.trunc)

    def test_math_copysign(self):
        self.binary_template_float32(math_copysign, np.copysign, start=-1)
        self.binary_template_float64(math_copysign, np.copysign, start=-1)

    def test_math_modf(self):

        def modf_template_nan(dtype, arytype):
            A = np.array([np.nan], dtype=dtype)
            B = np.zeros_like(A)
            C = np.zeros_like(A)
            cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
            cfunc[1, len(A)](A, B, C)
            self.assertTrue(np.isnan(B))
            self.assertTrue(np.isnan(C))

        def modf_template_compare(A, dtype, arytype):
            A = A.astype(dtype)
            B = np.zeros_like(A)
            C = np.zeros_like(A)
            cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
            cfunc[1, len(A)](A, B, C)
            D, E = np.modf(A)
            self.assertTrue(np.array_equal(B, D))
            self.assertTrue(np.array_equal(C, E))
        nelem = 50
        with self.subTest('float32 modf on simple float'):
            modf_template_compare(np.linspace(0, 10, nelem), dtype=np.float32, arytype=float32[:])
        with self.subTest('float32 modf on +- infinity'):
            modf_template_compare(np.array([np.inf, -np.inf]), dtype=np.float32, arytype=float32[:])
        with self.subTest('float32 modf on nan'):
            modf_template_nan(dtype=np.float32, arytype=float32[:])
        with self.subTest('float64 modf on simple float'):
            modf_template_compare(np.linspace(0, 10, nelem), dtype=np.float64, arytype=float64[:])
        with self.subTest('float64 modf on +- infinity'):
            modf_template_compare(np.array([np.inf, -np.inf]), dtype=np.float64, arytype=float64[:])
        with self.subTest('float64 modf on nan'):
            modf_template_nan(dtype=np.float64, arytype=float64[:])

    def test_math_fmod(self):
        self.binary_template_float32(math_fmod, np.fmod, start=1)
        self.binary_template_float64(math_fmod, np.fmod, start=1)

    def test_math_mod_binop(self):
        self.binary_template_float32(math_mod_binop, np.fmod, start=1)
        self.binary_template_float64(math_mod_binop, np.fmod, start=1)

    def test_math_isnan(self):
        self.unary_bool_template_float32(math_isnan, np.isnan)
        self.unary_bool_template_float64(math_isnan, np.isnan)
        self.unary_bool_template_int32(math_isnan, np.isnan)
        self.unary_bool_template_int64(math_isnan, np.isnan)
        self.unary_bool_special_values_float32(math_isnan, np.isnan)
        self.unary_bool_special_values_float64(math_isnan, np.isnan)

    def test_math_isinf(self):
        self.unary_bool_template_float32(math_isinf, np.isinf)
        self.unary_bool_template_float64(math_isinf, np.isinf)
        self.unary_bool_template_int32(math_isinf, np.isinf)
        self.unary_bool_template_int64(math_isinf, np.isinf)
        self.unary_bool_special_values_float32(math_isinf, np.isinf)
        self.unary_bool_special_values_float64(math_isinf, np.isinf)

    def test_math_isfinite(self):
        self.unary_bool_template_float32(math_isfinite, np.isfinite)
        self.unary_bool_template_float64(math_isfinite, np.isfinite)
        self.unary_bool_template_int32(math_isfinite, np.isfinite)
        self.unary_bool_template_int64(math_isfinite, np.isfinite)
        self.unary_bool_special_values_float32(math_isfinite, np.isfinite)
        self.unary_bool_special_values_float64(math_isfinite, np.isfinite)

    def test_math_degrees(self):
        self.unary_bool_template_float32(math_degrees, np.degrees)
        self.unary_bool_template_float64(math_degrees, np.degrees)

    def test_math_radians(self):
        self.unary_bool_template_float32(math_radians, np.radians)
        self.unary_bool_template_float64(math_radians, np.radians)