import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@skip_parfors_unsupported
class TestParfors(TestParforsBase):
    """ Tests cpython, reduction and various parfors features"""

    def test_arraymap(self):

        def test_impl(a, x, y):
            return a * x + y
        self.check_variants(test_impl, lambda: self.gen_linspace_variants(3))

    def test_0d_broadcast(self):

        def test_impl():
            X = np.array(1)
            Y = np.ones((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertEqual(countParfors(test_impl, ()), 1)

    def test_2d_parfor(self):

        def test_impl():
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertEqual(countParfors(test_impl, ()), 1)

    def test_nd_parfor(self):

        def case1():
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            yield (X, Y)
        data_gen = lambda: chain(case1(), self.gen_linspace_variants(2))

        def test_impl(X, Y):
            return np.sum(X + Y)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_np_func_direct_import(self):
        from numpy import ones

        def test_impl(n):
            A = ones(n)
            return A[0]
        n = 111
        self.check(test_impl, n)

    def test_size_assertion(self):

        def test_impl(m, n):
            A = np.ones(m)
            B = np.ones(n)
            return np.sum(A + B)
        self.check(test_impl, 10, 10)
        with self.assertRaises(AssertionError) as raises:
            cfunc = njit(parallel=True)(test_impl)
            cfunc(10, 9)
        msg = 'Sizes of A, B do not match'
        self.assertIn(msg, str(raises.exception))

    def test_cfg(self):

        def test_impl(x, is_positive, N):
            for i in numba.prange(2):
                for j in range(i * N // 2, (i + 1) * N // 2):
                    is_positive[j] = 0
                    if x[j] > 0:
                        is_positive[j] = 1
            return is_positive
        N = 100
        x = np.random.rand(N)
        is_positive = np.zeros(N)
        self.check(test_impl, x, is_positive, N)

    def test_reduce(self):

        def test_impl(A):
            init_val = 10
            return reduce(lambda a, b: min(a, b), A, init_val)
        n = 211
        A = np.random.ranf(n)
        self.check(test_impl, A)
        A = np.random.randint(10, size=n).astype(np.int32)
        self.check(test_impl, A)

        def test_impl():
            g = lambda x: x ** 2
            return reduce(g, np.array([1, 2, 3, 4, 5]), 2)
        with self.assertTypingError():
            self.check(test_impl)
        n = 160
        A = np.random.randint(10, size=n).astype(np.int32)

        def test_impl(A):
            return np.sum(A[A >= 3])
        self.check(test_impl, A)

        def test_impl(A):
            B = A[:, 0]
            return np.sum(A[B >= 3, 1])
        self.check(test_impl, A.reshape((16, 10)))

        def test_impl(A):
            B = A[:, 0]
            return np.sum(A[B >= 3, 1:2])
        self.check(test_impl, A.reshape((16, 10)))
        self.assertEqual(countParfors(test_impl, (numba.float64[:, :],)), 2)

        def test_impl(A):
            min_val = np.amin(A)
            return A - min_val
        self.check(test_impl, A)
        self.assertEqual(countParfors(test_impl, (numba.float64[:],)), 2)

    def test_use_of_reduction_var1(self):

        def test_impl():
            acc = 0
            for i in prange(1):
                acc = cmath.sqrt(acc)
            return acc
        msg = 'Use of reduction variable acc in an unsupported reduction function.'
        with self.assertRaises(ValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_unsupported_floordiv1(self):

        def test_impl():
            acc = 100
            for i in prange(2):
                acc //= 2
            return acc
        msg = 'Parallel floordiv reductions are not supported. If all divisors are integers then a floordiv reduction can in some cases be parallelized as a multiply reduction followed by a floordiv of the resulting product.'
        with self.assertRaises(errors.NumbaValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_unsupported_xor1(self):

        def test_impl():
            acc = 100
            for i in prange(2):
                acc ^= i + 2
            return acc
        msg = 'Use of reduction variable acc in an unsupported reduction function.'
        with self.assertRaises(ValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_parfor_array_access1(self):

        def test_impl(n):
            A = np.ones(n)
            return A.sum()
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    def test_parfor_array_access2(self):

        def test_impl(n):
            A = np.ones(n)
            m = 0
            n = 0
            for i in numba.prange(len(A)):
                m += A[i]
            for i in numba.prange(len(A)):
                if m == n:
                    n += A[i]
            return m + n
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    def test_parfor_array_access3(self):

        def test_impl(n):
            A = np.ones(n, np.int64)
            m = 0
            for i in numba.prange(len(A)):
                m += A[i]
                if m == 2:
                    i = m
        n = 211
        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.check(test_impl, n)
        self.assertIn('Overwrite of parallel loop index', str(raises.exception))

    @needs_blas
    def test_parfor_array_access4(self):

        def test_impl(A, b):
            return np.dot(A, b)
        n = 211
        d = 4
        A = np.random.ranf((n, d))
        b = np.random.ranf(d)
        self.check(test_impl, A, b)
        test_ir, tp = get_optimized_numba_ir(test_impl, (types.Array(types.float64, 2, 'C'), types.Array(types.float64, 1, 'C')))
        self.assertTrue(len(test_ir.blocks) == 1 and 0 in test_ir.blocks)
        block = test_ir.blocks[0]
        parfor_found = False
        parfor = None
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                parfor_found = True
                parfor = stmt
        self.assertTrue(parfor_found)
        build_tuple_found = False
        for bl in parfor.loop_body.values():
            for stmt in bl.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'build_tuple'):
                    build_tuple_found = True
                    self.assertTrue(parfor.index_var in stmt.value.items)
        self.assertTrue(build_tuple_found)

    def test_parfor_dtype_type(self):

        def test_impl(a):
            for i in numba.prange(len(a)):
                a[i] = a.dtype.type(0)
            return a[4]
        a = np.ones(10)
        self.check(test_impl, a)

    def test_parfor_array_access5(self):

        def test_impl(n):
            X = np.ones((n, 3))
            y = 0
            for i in numba.prange(n):
                y += X[i, :].sum()
            return y
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    @disabled_test
    def test_parfor_hoist_setitem(self):

        def test_impl(out):
            for i in prange(10):
                out[0] = 2 * out[0]
            return out[0]
        out = np.ones(1)
        self.check(test_impl, out)

    @needs_blas
    def test_parfor_generate_fuse(self):

        def test_impl(N, D):
            w = np.ones(D)
            X = np.ones((N, D))
            Y = np.ones(N)
            for i in range(3):
                B = -Y * np.dot(X, w)
            return B
        n = 211
        d = 3
        self.check(test_impl, n, d)
        self.assertEqual(countArrayAllocs(test_impl, (types.intp, types.intp)), 4)
        self.assertEqual(countParfors(test_impl, (types.intp, types.intp)), 4)

    def test_ufunc_expr(self):

        def test_impl(A, B):
            return np.bitwise_and(A, B)
        A = np.ones(3, np.uint8)
        B = np.ones(3, np.uint8)
        B[1] = 0
        self.check(test_impl, A, B)

    def test_find_callname_intrinsic(self):

        def test_impl(n):
            A = unsafe_empty((n,))
            for i in range(n):
                A[i] = i + 2.0
            return A
        self.assertEqual(countArrayAllocs(test_impl, (types.intp,)), 1)

    def test_reduction_var_reuse(self):

        def test_impl(n):
            acc = 0
            for i in prange(n):
                acc += 1
            for i in prange(n):
                acc += 2
            return acc
        self.check(test_impl, 16)

    def test_non_identity_initial(self):

        def test_impl(A, cond):
            s = 1
            for i in prange(A.shape[0]):
                if cond[i]:
                    s += 1
            return s
        self.check(test_impl, np.ones(10), np.ones(10).astype('bool'))

    def test_if_not_else_reduction(self):

        def test_impl(A, cond):
            s = 1
            t = 10
            for i in prange(A.shape[0]):
                if cond[i]:
                    s += 1
                    t += 1
                else:
                    s += 2
            return s + t
        self.check(test_impl, np.ones(10), np.ones(10).astype('bool'))

    def test_two_d_array_reduction_reuse(self):

        def test_impl(n):
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_one_d_array_reduction(self):

        def test_impl(n):
            result = np.zeros(1, np.int_)
            for i in numba.prange(n):
                result += np.array([i], np.int_)
            return result
        self.check(test_impl, 100)

    def test_two_d_array_reduction(self):

        def test_impl(n):
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_two_d_array_reduction_with_float_sizes(self):

        def test_impl(n):
            shp = (2, 3)
            result1 = np.zeros(shp, np.float32)
            tmp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_two_d_array_reduction_prod(self):

        def test_impl(n):
            shp = (13, 17)
            result1 = 2 * np.ones(shp, np.int_)
            tmp = 2 * np.ones_like(result1)
            for i in numba.prange(n):
                result1 *= tmp
            return result1
        self.check(test_impl, 100)

    def test_three_d_array_reduction(self):

        def test_impl(n):
            shp = (3, 2, 7)
            result1 = np.zeros(shp, np.int_)
            for i in numba.prange(n):
                result1 += np.ones(shp, np.int_)
            return result1
        self.check(test_impl, 100)

    def test_preparfor_canonicalize_kws(self):

        def test_impl(A):
            return A.argsort() + 1
        n = 211
        A = np.arange(n)
        self.check(test_impl, A)

    def test_preparfor_datetime64(self):

        def test_impl(A):
            return A.dtype
        A = np.empty(1, np.dtype('datetime64[ns]'))
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        self.assertEqual(cpfunc.entry_point(A), test_impl(A))

    def test_no_hoisting_with_member_function_call(self):

        def test_impl(X):
            n = X.shape[0]
            acc = 0
            for i in prange(n):
                R = {1, 2, 3}
                R.add(i)
                tmp = 0
                for x in R:
                    tmp += x
                acc += tmp
            return acc
        self.check(test_impl, np.random.ranf(128))

    def test_array_compare_scalar(self):
        """ issue3671: X != 0 becomes an arrayexpr with operator.ne.
            That is turned into a parfor by devectorizing.  Make sure
            the return type of the devectorized operator.ne
            on integer types works properly.
        """

        def test_impl():
            X = np.zeros(10, dtype=np.int_)
            return X != 0
        self.check(test_impl)

    def test_array_analysis_optional_def(self):

        def test_impl(x, half):
            size = len(x)
            parr = x[0:size]
            if half:
                parr = x[0:size // 2]
            return parr.sum()
        x = np.ones(20)
        self.check(test_impl, x, True, check_scheduling=False)

    def test_prange_side_effects(self):

        def test_impl(a, b):
            data = np.empty(len(a), dtype=np.float64)
            size = len(data)
            for i in numba.prange(size):
                data[i] = a[i]
            for i in numba.prange(size):
                data[i] = data[i] + b[i]
            return data
        x = np.arange(10 ** 2, dtype=float)
        y = np.arange(10 ** 2, dtype=float)
        self.check(test_impl, x, y)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), types.Array(types.float64, 1, 'C'))), 1)

    def test_tuple1(self):

        def test_impl(a):
            atup = (3, 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + atup[1] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_tuple2(self):

        def test_impl(a):
            atup = a.shape
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_tuple3(self):

        def test_impl(a):
            atup = (np.arange(10), 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0][5] + atup[1] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple1(self):

        def test_impl(a):
            antup = TestNamedTuple(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple2(self):
        TestNamedTuple2 = namedtuple('TestNamedTuple2', ('part0', 'part1'))

        def test_impl(a):
            antup = TestNamedTuple2(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple3(self):
        TestNamedTuple3 = namedtuple(f'TestNamedTuple3', ['y'])

        def test_impl(a):
            a.y[:] = 5

        def comparer(a, b):
            np.testing.assert_almost_equal(a.y, b.y)
        x = TestNamedTuple3(y=np.zeros(10))
        self.check(test_impl, x, check_arg_equality=[comparer])

    def test_inplace_binop(self):

        def test_impl(a, b):
            b += a
            return b
        X = np.arange(10) + 10
        Y = np.arange(10) + 100
        self.check(test_impl, X, Y)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), types.Array(types.float64, 1, 'C'))), 1)

    def test_tuple_concat(self):

        def test_impl(a):
            n = len(a)
            array_shape = (n, n)
            indices = np.zeros((1,) + array_shape + (1,), dtype=np.uint64)
            k_list = indices[0, :]
            for i, g in enumerate(a):
                k_list[i, i] = i
            return k_list
        x = np.array([1, 1])
        self.check(test_impl, x)

    def test_tuple_concat_with_reverse_slice(self):

        def test_impl(a):
            n = len(a)
            array_shape = (n, n)
            indices = np.zeros(((1,) + array_shape + (1,))[:-1], dtype=np.uint64)
            k_list = indices[0, :]
            for i, g in enumerate(a):
                k_list[i, i] = i
            return k_list
        x = np.array([1, 1])
        self.check(test_impl, x)

    def test_array_tuple_concat(self):

        def test_impl(a):
            S = (a,) + (a, a)
            return S[0].sum()
        x = np.ones((3, 3))
        self.check(test_impl, x)

    def test_high_dimension1(self):

        def test_impl(x):
            return x * 5.0
        x = np.ones((2, 2, 2, 2, 2, 15))
        self.check(test_impl, x)

    def test_tuple_arg(self):

        def test_impl(x, sz):
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.empty(sz), sz)

    def test_tuple_arg_not_whole_array(self):

        def test_impl(x, sz):
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz), (10, 3))

    def test_tuple_for_pndindex(self):

        def test_impl(x):
            sz = (10, 5)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz))

    def test_tuple_arg_literal(self):

        def test_impl(x, first):
            sz = (first, 5)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz), 10)

    def test_tuple_of_literal_nonliteral(self):

        def test_impl(x, sz):
            for i in numba.pndindex(sz):
                x[i] = 1
            return x

        def call(x, fn):
            return fn(x, (10, 3))
        get_input = lambda: np.zeros((10, 10))
        expected = call(get_input(), test_impl)

        def check(dec):
            f1 = dec(test_impl)
            f2 = njit(call)
            got = f2(get_input(), f1)
            self.assertPreciseEqual(expected, got)
        for d in (njit, njit(parallel=True)):
            check(d)

    def test_tuple_arg_1d(self):

        def test_impl(x, sz):
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10,)
        self.check(test_impl, np.zeros(sz), sz)

    def test_tuple_arg_1d_literal(self):

        def test_impl(x):
            sz = (10,)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10,)
        self.check(test_impl, np.zeros(sz))

    def test_int_arg_pndindex(self):

        def test_impl(x, sz):
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        self.check(test_impl, np.zeros((10, 10)), 3)

    def test_prange_unknown_call1(self):

        @register_jitable
        def issue7854_proc(u, i, even, size):
            for j in range((even + i + 1) % 2 + 1, size - 1, 2):
                u[i, j] = u[i + 1, j] + 1

        def test_impl(u, size):
            for i in numba.prange(1, size - 1):
                issue7854_proc(u, i, 0, size)
            for i in numba.prange(1, size - 1):
                issue7854_proc(u, i, 1, size)
            return u
        size = 4
        u = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, u, size)

    def test_prange_index_calc1(self):

        def test_impl(u, size):
            for i in numba.prange(1, size - 1):
                for j in range((i + 1) % 2 + 1, size - 1, 2):
                    u[i, j] = u[i + 1, j] + 1
            for i in numba.prange(1, size - 1):
                for j in range(i % 2 + 1, size - 1, 2):
                    u[i, j] = u[i + 1, j] + 1
            return u
        size = 4
        u = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, u, size)

    def test_prange_reverse_order1(self):

        def test_impl(a, b, size):
            for i in numba.prange(size):
                for j in range(size):
                    a[i, j] = b[i, j] + 1
            for i in numba.prange(size):
                for j in range(size):
                    b[j, i] = 3
            return a[0, 0] + b[0, 0]
        size = 10
        a = np.zeros((size, size))
        b = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, b, size)

    def test_prange_parfor_index_then_not(self):

        def test_impl(a, size):
            b = 0
            for i in numba.prange(size):
                a[i] = i
            for i in numba.prange(size):
                b += a[5]
            return b
        size = 10
        a = np.zeros(size)
        cptypes = (numba.float64[:], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, size)

    def test_prange_parfor_index_const_tuple_fusion(self):

        def test_impl(a, tup, size):
            acc = 0
            for i in numba.prange(size):
                a[i] = i + tup[i]
            for i in numba.prange(size):
                acc += a[i] + tup[1]
            return acc
        size = 10
        a = np.zeros(size)
        b = tuple(a)
        cptypes = (numba.float64[:], types.containers.UniTuple(types.float64, size), types.intp)
        self.assertEqual(countParfors(test_impl, cptypes), 1)
        self.check(test_impl, a, b, size)

    def test_prange_non_parfor_index_then_opposite(self):

        def test_impl(a, b, size):
            for i in numba.prange(size):
                b[i] = a[5]
            for i in numba.prange(size):
                a[i] = i
            b[0] += a[0]
            return b
        size = 10
        a = np.zeros(size)
        b = np.zeros(size)
        cptypes = (numba.float64[:], numba.float64[:], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, b, size)

    def test_prange_optional(self):

        def test_impl(arr, pred=None):
            for i in prange(1):
                if pred is not None:
                    arr[i] = 0.0
        arr = np.ones(10)
        self.check(test_impl, arr, None, check_arg_equality=[np.testing.assert_almost_equal, lambda x, y: x == y])
        self.assertEqual(arr.sum(), 10.0)

    def test_untraced_value_tuple(self):

        def test_impl():
            a = (1.2, 1.3)
            return a[0]
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_recursive_untraced_value_tuple(self):

        def test_impl():
            a = ((1.2, 1.3),)
            return a[0][0]
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_untraced_value_parfor(self):

        def test_impl(arr):
            a = (1.2, 1.3)
            n1 = len(arr)
            arr2 = np.empty(n1, np.float64)
            for i in prange(n1):
                arr2[i] = arr[i] * a[0]
            n2 = len(arr2)
            arr3 = np.empty(n2, np.float64)
            for j in prange(n2):
                arr3[j] = arr2[j] - a[1]
            total = 0.0
            n3 = len(arr3)
            for k in prange(n3):
                total += arr3[k]
            return total + a[0]
        arg = (types.Array(types.int64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, arg), 1)
        arr = np.arange(10, dtype=np.int64)
        self.check(test_impl, arr)

    def test_setitem_2d_one_replaced(self):

        def test_impl(x):
            count = 0
            for n in range(x.shape[0]):
                if n:
                    n
                x[count, :] = 1
                count += 1
            return x
        self.check(test_impl, np.zeros((3, 1)))

    def test_1array_control_flow(self):

        def test_impl(arr, flag1, flag2):
            inv = np.arange(arr.size)
            if flag1:
                return inv.astype(np.float64)
            if flag2:
                ret = inv[inv]
            else:
                ret = inv[inv - 1]
            return ret / arr.size
        arr = np.arange(100)
        self.check(test_impl, arr, True, False)
        self.check(test_impl, arr, True, True)
        self.check(test_impl, arr, False, False)

    def test_2array_1_control_flow(self):

        def test_impl(arr, l, flag):
            inv1 = np.arange(arr.size)
            inv2 = np.arange(l, arr.size + l)
            if flag:
                ret = inv1[inv1]
            else:
                ret = inv1[inv1 - 1]
            return ret / inv2
        arr = np.arange(100)
        self.check(test_impl, arr, 10, True)
        self.check(test_impl, arr, 10, False)

    def test_2array_2_control_flow(self):

        def test_impl(arr, l, flag):
            inv1 = np.arange(arr.size)
            inv2 = np.arange(l, arr.size + l)
            if flag:
                ret1 = inv1[inv1]
                ret2 = inv2[inv1]
            else:
                ret1 = inv1[inv1 - 1]
                ret2 = inv2[inv1 - 1]
            return ret1 / ret2
        arr = np.arange(100)
        self.check(test_impl, arr, 10, True)
        self.check(test_impl, arr, 10, False)

    def test_issue8515(self):

        def test_impl(n):
            r = np.zeros(n, dtype=np.intp)
            c = np.zeros(n, dtype=np.intp)
            for i in prange(n):
                for j in range(i):
                    c[i] += 1
            for i in prange(n):
                if i == 0:
                    continue
                r[i] = c[i] - c[i - 1]
            return r[1:]
        self.check(test_impl, 15)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 2)

    def test_issue9029(self):

        def test_impl(i1, i2):
            N = 30
            S = 3
            a = np.empty((N, N))
            for y in range(N):
                for x in range(N):
                    values = np.ones(S)
                    v = values[0]
                    p2 = np.empty(S)
                    for i in prange(i1, i2):
                        p2[i] = 1
                    j = p2[0]
                    a[y, x] = v + j
            return a
        self.check(test_impl, 0, 3)

    def test_fusion_no_side_effects(self):

        def test_impl(a, b):
            X = np.ones(100)
            b = math.ceil(b)
            Y = np.ones(100)
            c = int(max(a, b))
            return X + Y + c
        self.check(test_impl, 3.7, 4.3)
        self.assertEqual(countParfors(test_impl, (types.float64, types.float64)), 1)

    def test_issue9256_lower_sroa_conflict(self):

        @njit(parallel=True)
        def def_in_loop(x):
            c = 0
            set_num_threads(1)
            for i in prange(x):
                c = i
            return c
        self.assertEqual(def_in_loop(10), def_in_loop.py_func(10))

    def test_issue9256_lower_sroa_conflict_variant1(self):

        def def_in_loop(x):
            c = x
            set_num_threads(1)
            for _i in prange(x):
                if c:
                    d = x + 4
            return (c, d > 0)
        expected = def_in_loop(4)
        self.assertEqual(expected, njit(parallel=False)(def_in_loop)(4))
        self.assertEqual(expected, njit(parallel=True)(def_in_loop)(4))

    def test_issue9256_lower_sroa_conflict_variant2(self):

        def def_in_loop(x):
            c = x
            set_num_threads(1)
            for _i in prange(x):
                if c:
                    for _j in range(x):
                        d = x + 4
            return (c, d > 0)
        expected = def_in_loop(4)
        self.assertEqual(expected, njit(parallel=False)(def_in_loop)(4))
        self.assertEqual(expected, njit(parallel=True)(def_in_loop)(4))