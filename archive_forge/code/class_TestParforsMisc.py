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
class TestParforsMisc(TestParforsBase):
    """
    Tests miscellaneous parts of ParallelAccelerator use.
    """

    def test_no_warn_if_cache_set(self):

        def pyfunc():
            arr = np.ones(100)
            for i in prange(arr.size):
                arr[i] += i
            return arr
        cfunc = njit(parallel=True, cache=True)(pyfunc)
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            warnings.filterwarnings(action='ignore', module='typeguard')
            warnings.filterwarnings(action='ignore', message='.*TBB_INTERFACE_VERSION.*', category=numba.errors.NumbaWarning, module='numba\\.np\\.ufunc\\.parallel.*')
            cfunc()
        self.assertEqual(len(raised_warnings), 0)
        has_dynamic_globals = [cres.library.has_dynamic_globals for cres in cfunc.overloads.values()]
        self.assertEqual(has_dynamic_globals, [False])

    def test_statement_reordering_respects_aliasing(self):

        def impl():
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            print('a[3]:', a[3])
            print('a[3]:', a[3])
            return a
        cres = self.compile_parallel(impl, ())
        with captured_stdout() as stdout:
            cres.entry_point()
        for line in stdout.getvalue().splitlines():
            self.assertEqual('a[3]: 2.0', line)

    def test_parfor_ufunc_typing(self):

        def test_impl(A):
            return np.isinf(A)
        A = np.array([np.inf, 0.0])
        cfunc = njit(parallel=True)(test_impl)
        old_seq_flag = numba.parfors.parfor.sequential_parfor_lowering
        try:
            numba.parfors.parfor.sequential_parfor_lowering = True
            np.testing.assert_array_equal(test_impl(A), cfunc(A))
        finally:
            numba.parfors.parfor.sequential_parfor_lowering = old_seq_flag

    def test_init_block_dce(self):

        def test_impl():
            res = 0
            arr = [1, 2, 3, 4, 5]
            numba.parfors.parfor.init_prange()
            dummy = arr
            for i in numba.prange(5):
                res += arr[i]
            return res + dummy[2]
        self.assertEqual(get_init_block_size(test_impl, ()), 0)

    def test_alias_analysis_for_parfor1(self):

        def test_impl():
            acc = 0
            for _ in range(4):
                acc += 1
            data = np.zeros((acc,))
            return data
        self.check(test_impl)

    def test_no_state_change_in_gufunc_lowering_on_error(self):
        BROKEN_MSG = 'BROKEN_MSG'

        @register_pass(mutates_CFG=True, analysis_only=False)
        class BreakParfors(AnalysisPass):
            _name = 'break_parfors'

            def __init__(self):
                AnalysisPass.__init__(self)

            def run_pass(self, state):
                for blk in state.func_ir.blocks.values():
                    for stmt in blk.body:
                        if isinstance(stmt, numba.parfors.parfor.Parfor):

                            class Broken(list):

                                def difference(self, other):
                                    raise errors.LoweringError(BROKEN_MSG)
                            stmt.races = Broken()
                    return True

        class BreakParforsCompiler(CompilerBase):

            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(BreakParfors, IRLegalization)
                pm.finalize()
                return [pm]

        @njit(parallel=True, pipeline_class=BreakParforsCompiler)
        def foo():
            x = 1
            for _ in prange(1):
                x += 1
            return x
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
        with self.assertRaises(errors.LoweringError) as raises:
            foo()
        self.assertIn(BROKEN_MSG, str(raises.exception))
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)

    def test_issue_5098(self):

        class DummyType(types.Opaque):
            pass
        dummy_type = DummyType('my_dummy')
        register_model(DummyType)(models.OpaqueModel)

        class Dummy(object):
            pass

        @typeof_impl.register(Dummy)
        def typeof_Dummy(val, c):
            return dummy_type

        @unbox(DummyType)
        def unbox_index(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())

        @overload_method(DummyType, 'method1', jit_options={'parallel': True})
        def _get_method1(obj, arr, func):

            def _foo(obj, arr, func):

                def baz(a, f):
                    c = a.copy()
                    c[np.isinf(a)] = np.nan
                    return f(c)
                length = len(arr)
                output_arr = np.empty(length, dtype=np.float64)
                for i in prange(length):
                    output_arr[i] = baz(arr[i], func)
                for i in prange(length - 1):
                    output_arr[i] += baz(arr[i], func)
                return output_arr
            return _foo

        @njit
        def bar(v):
            return v.mean()

        @njit
        def test1(d):
            return d.method1(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), bar)
        save_state = numba.parfors.parfor.sequential_parfor_lowering
        self.assertFalse(save_state)
        try:
            test1(Dummy())
            self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
        finally:
            numba.parfors.parfor.sequential_parfor_lowering = save_state

    def test_oversized_tuple_as_arg_to_kernel(self):

        @njit(parallel=True)
        def oversize_tuple(idx):
            big_tup = (1, 2, 3, 4)
            z = 0
            for x in prange(10):
                z += big_tup[idx]
            return z
        with override_env_config('NUMBA_PARFOR_MAX_TUPLE_SIZE', '3'):
            with self.assertRaises(errors.UnsupportedParforsError) as raises:
                oversize_tuple(0)
        errstr = str(raises.exception)
        self.assertIn('Use of a tuple', errstr)
        self.assertIn('in a parallel region', errstr)

    def test_issue5167(self):

        def ndvi_njit(img_nir, img_red):
            fillvalue = 0
            out_img = np.full(img_nir.shape, fillvalue, dtype=img_nir.dtype)
            dims = img_nir.shape
            for y in prange(dims[0]):
                for x in prange(dims[1]):
                    out_img[y, x] = (img_nir[y, x] - img_red[y, x]) / (img_nir[y, x] + img_red[y, x])
            return out_img
        tile_shape = (4, 4)
        array1 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        array2 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        self.check(ndvi_njit, array1, array2)

    def test_issue5065(self):

        def reproducer(a, dist, dist_args):
            result = np.zeros((a.shape[0], a.shape[0]), dtype=np.float32)
            for i in prange(a.shape[0]):
                for j in range(i + 1, a.shape[0]):
                    d = dist(a[i], a[j], *dist_args)
                    result[i, j] = d
                    result[j, i] = d
            return result

        @njit
        def euclidean(x, y):
            result = 0.0
            for i in range(x.shape[0]):
                result += (x[i] - y[i]) ** 2
            return np.sqrt(result)
        a = np.random.random(size=(5, 2))
        got = njit(parallel=True)(reproducer)(a.copy(), euclidean, ())
        expected = reproducer(a.copy(), euclidean, ())
        np.testing.assert_allclose(got, expected)

    def test_issue5001(self):

        def test_numba_parallel(myarray):
            result = [0] * len(myarray)
            for i in prange(len(myarray)):
                result[i] = len(myarray[i])
            return result
        myarray = (np.empty(100), np.empty(50))
        self.check(test_numba_parallel, myarray)

    def test_issue3169(self):

        @njit
        def foo(grids):
            pass

        @njit(parallel=True)
        def bar(grids):
            for x in prange(1):
                foo(grids)
        bar(([1],) * 2)

    @disabled_test
    def test_issue4846(self):
        mytype = namedtuple('mytype', ('a', 'b'))

        def outer(mydata):
            for k in prange(3):
                inner(k, mydata)
            return mydata.a

        @njit(nogil=True)
        def inner(k, mydata):
            f = (k, mydata.a)
            g = (k, mydata.b)
        mydata = mytype(a='a', b='b')
        self.check(outer, mydata)

    def test_issue3748(self):

        def test1b():
            x = (1, 2, 3, 4, 5)
            a = 0
            for i in prange(len(x)):
                a += x[i]
            return a
        self.check(test1b)

    def test_issue5277(self):

        def parallel_test(size, arr):
            for x in prange(size[0]):
                for y in prange(size[1]):
                    arr[y][x] = x * 4.5 + y
            return arr
        size = (10, 10)
        arr = np.zeros(size, dtype=int)
        self.check(parallel_test, size, arr)

    def test_issue5570_ssa_races(self):

        @njit(parallel=True)
        def foo(src, method, out):
            for i in prange(1):
                for j in range(1):
                    out[i, j] = 1
            if method:
                out += 1
            return out
        src = np.zeros((5, 5))
        method = 57
        out = np.zeros((2, 2))
        self.assertPreciseEqual(foo(src, method, out), foo.py_func(src, method, out))

    def test_issue6095_numpy_max(self):

        @njit(parallel=True)
        def find_maxima_3D_jit(args):
            package = args
            for index in range(0, 10):
                z_stack = package[index, :, :]
            return np.max(z_stack)
        np.random.seed(0)
        args = np.random.random((10, 10, 10))
        self.assertPreciseEqual(find_maxima_3D_jit(args), find_maxima_3D_jit.py_func(args))

    def test_issue5942_1(self):

        def test_impl(gg, gg_next):
            gs = gg.shape
            d = gs[0]
            for i_gg in prange(d):
                gg_next[i_gg, :] = gg[i_gg, :]
                gg_next[i_gg, 0] += 1
            return gg_next
        d = 4
        k = 2
        gg = np.zeros((d, k), dtype=np.int32)
        gg_next = np.zeros((d, k), dtype=np.int32)
        self.check(test_impl, gg, gg_next)

    def test_issue5942_2(self):

        def test_impl(d, k):
            gg = np.zeros((d, k), dtype=np.int32)
            gg_next = np.zeros((d, k), dtype=np.int32)
            for i_gg in prange(d):
                for n in range(k):
                    gg[i_gg, n] = i_gg
                gg_next[i_gg, :] = gg[i_gg, :]
                gg_next[i_gg, 0] += 1
            return gg_next
        d = 4
        k = 2
        self.check(test_impl, d, k)

    @skip_unless_scipy
    def test_issue6102(self):

        @njit(parallel=True)
        def f(r):
            for ir in prange(r.shape[0]):
                dist = np.inf
                tr = np.array([0, 0, 0], dtype=np.float32)
                for i in [1, 0, -1]:
                    dist_t = np.linalg.norm(r[ir, :] + i)
                    if dist_t < dist:
                        dist = dist_t
                        tr = np.array([i, i, i], dtype=np.float32)
                r[ir, :] += tr
            return r
        r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertPreciseEqual(f(r), f.py_func(r))

    def test_issue6774(self):

        def test_impl():
            n = 5
            na_mask = np.ones((n,))
            result = np.empty((n - 1,))
            for i in prange(len(result)):
                result[i] = np.sum(na_mask[i:i + 1])
            return result
        self.check(test_impl)

    def test_issue4963_globals(self):

        def test_impl():
            buf = np.zeros((_GLOBAL_INT_FOR_TESTING1, _GLOBAL_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    def test_issue4963_freevars(self):
        _FREEVAR_INT_FOR_TESTING1 = 17
        _FREEVAR_INT_FOR_TESTING2 = 5

        def test_impl():
            buf = np.zeros((_FREEVAR_INT_FOR_TESTING1, _FREEVAR_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    def test_issue_9182_recursion_error(self):
        from numba.types import ListType, Tuple, intp

        @numba.njit
        def _sink(x):
            pass

        @numba.njit(cache=False, parallel=True)
        def _ground_node_rule(clauses, nodes):
            for piter in prange(len(nodes)):
                for clause in clauses:
                    clause_type = clause[0]
                    clause_variables = clause[2]
                    if clause_type == 0:
                        clause_var_1 = clause_variables[0]
                    elif len(clause_variables) == 2:
                        clause_var_1, clause_var_2 = (clause_variables[0], clause_variables[1])
                    elif len(clause_variables) == 4:
                        pass
                    if clause_type == 1:
                        _sink(clause_var_1)
                        _sink(clause_var_2)
        _ground_node_rule.compile((ListType(Tuple([intp, intp, ListType(intp)])), ListType(intp)))

    def test_lookup_cycle_detection(self):

        @njit(parallel=True)
        def foo():
            acc = 0
            for n in prange(1):
                for i in range(1):
                    for j in range(1):
                        acc += 1
            return acc
        self.assertEqual(foo(), foo.py_func())