import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
class TestReportedSSAIssues(SSABaseTest):

    def test_issue2194(self):

        @njit
        def foo():
            V = np.empty(1)
            s = np.uint32(1)
            for i in range(s):
                V[i] = 1
            for i in range(s, 1):
                pass
        self.check_func(foo)

    def test_issue3094(self):

        @njit
        def doit(x):
            return x

        @njit
        def foo(pred):
            if pred:
                x = True
            else:
                x = False
            return doit(x)
        self.check_func(foo, False)

    def test_issue3931(self):

        @njit
        def foo(arr):
            for i in range(1):
                arr = arr.reshape(3 * 2)
                arr = arr.reshape(3, 2)
            return arr
        np.testing.assert_allclose(foo(np.zeros((3, 2))), foo.py_func(np.zeros((3, 2))))

    def test_issue3976(self):

        def overload_this(a):
            return 'dummy'

        @njit
        def foo(a):
            if a:
                s = 5
                s = overload_this(s)
            else:
                s = 'b'
            return s

        @overload(overload_this)
        def ol(a):
            return overload_this
        self.check_func(foo, True)

    def test_issue3979(self):

        @njit
        def foo(A, B):
            x = A[0]
            y = B[0]
            for i in A:
                x = i
            for i in B:
                y = i
            return (x, y)
        self.check_func(foo, (1, 2), ('A', 'B'))

    def test_issue5219(self):

        def overload_this(a, b=None):
            if isinstance(b, tuple):
                b = b[0]
            return b

        @overload(overload_this)
        def ol(a, b=None):
            b_is_tuple = isinstance(b, (types.Tuple, types.UniTuple))

            def impl(a, b=None):
                if b_is_tuple is True:
                    b = b[0]
                return b
            return impl

        @njit
        def test_tuple(a, b):
            overload_this(a, b)
        self.check_func(test_tuple, 1, (2,))

    def test_issue5223(self):

        @njit
        def bar(x):
            if len(x) == 5:
                return x
            x = x.copy()
            for i in range(len(x)):
                x[i] += 1
            return x
        a = np.ones(5)
        a.flags.writeable = False
        np.testing.assert_allclose(bar(a), bar.py_func(a))

    def test_issue5243(self):

        @njit
        def foo(q):
            lin = np.array((0.1, 0.6, 0.3))
            stencil = np.zeros((3, 3))
            stencil[0, 0] = q[0, 0]
            return lin[0]
        self.check_func(foo, np.zeros((2, 2)))

    def test_issue5482_missing_variable_init(self):

        @njit('(intp, intp, intp)')
        def foo(x, v, n):
            for i in range(n):
                if i == 0:
                    if i == x:
                        pass
                    else:
                        problematic = v
                elif i == x:
                    pass
                else:
                    problematic = problematic + v
            return problematic

    def test_issue5482_objmode_expr_null_lowering(self):
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.untyped_passes import ReconstructSSA, IRProcessing
        from numba.core.typed_passes import PreLowerStripPhis

        class CustomPipeline(CompilerBase):

            def define_pipelines(self):
                pm = DefaultPassBuilder.define_objectmode_pipeline(self.state)
                pm.add_pass_after(ReconstructSSA, IRProcessing)
                pm.add_pass_after(PreLowerStripPhis, ReconstructSSA)
                pm.finalize()
                return [pm]

        @jit('(intp, intp, intp)', looplift=False, pipeline_class=CustomPipeline)
        def foo(x, v, n):
            for i in range(n):
                if i == n:
                    if i == x:
                        pass
                    else:
                        problematic = v
                elif i == x:
                    pass
                else:
                    problematic = problematic + v
            return problematic

    def test_issue5493_unneeded_phi(self):
        data = (np.ones(2), np.ones(2))
        A = np.ones(1)
        B = np.ones((1, 1))

        def foo(m, n, data):
            if len(data) == 1:
                v0 = data[0]
            else:
                v0 = data[0]
                for _ in range(1, len(data)):
                    v0 += A
            for t in range(1, m):
                for idx in range(n):
                    t = B
                    if idx == 0:
                        if idx == n - 1:
                            pass
                        else:
                            problematic = t
                    elif idx == n - 1:
                        pass
                    else:
                        problematic = problematic + t
            return problematic
        expect = foo(10, 10, data)
        res1 = njit(foo)(10, 10, data)
        res2 = jit(forceobj=True, looplift=False)(foo)(10, 10, data)
        np.testing.assert_array_equal(expect, res1)
        np.testing.assert_array_equal(expect, res2)

    def test_issue5623_equal_statements_in_same_bb(self):

        def foo(pred, stack):
            i = 0
            c = 1
            if pred is True:
                stack[i] = c
                i += 1
                stack[i] = c
                i += 1
        python = np.array([0, 666])
        foo(True, python)
        nb = np.array([0, 666])
        njit(foo)(True, nb)
        expect = np.array([1, 1])
        np.testing.assert_array_equal(python, expect)
        np.testing.assert_array_equal(nb, expect)

    def test_issue5678_non_minimal_phi(self):
        from numba.core.compiler import CompilerBase, DefaultPassBuilder
        from numba.core.untyped_passes import ReconstructSSA, FunctionPass, register_pass
        phi_counter = []

        @register_pass(mutates_CFG=False, analysis_only=True)
        class CheckSSAMinimal(FunctionPass):
            _name = self.__class__.__qualname__ + '.CheckSSAMinimal'

            def __init__(self):
                super().__init__(self)

            def run_pass(self, state):
                ct = 0
                for blk in state.func_ir.blocks.values():
                    ct += len(list(blk.find_exprs('phi')))
                phi_counter.append(ct)
                return True

        class CustomPipeline(CompilerBase):

            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(CheckSSAMinimal, ReconstructSSA)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=CustomPipeline)
        def while_for(n, max_iter=1):
            a = np.empty((n, n))
            i = 0
            while i <= max_iter:
                for j in range(len(a)):
                    for k in range(len(a)):
                        a[j, k] = j + k
                i += 1
            return a
        self.assertPreciseEqual(while_for(10), while_for.py_func(10))
        self.assertEqual(phi_counter, [1])

    def test_issue9242_use_not_dom_def(self):
        from numba.core.ir import FunctionIR
        from numba.core.compiler_machinery import AnalysisPass, register_pass

        def check(fir: FunctionIR):
            [blk, *_] = fir.blocks.values()
            var = blk.scope.get('d')
            defn = fir.get_definition(var)
            self.assertEqual(defn.op, 'phi')
            self.assertIn(ir.UNDEFINED, defn.incoming_values)

        @register_pass(mutates_CFG=False, analysis_only=True)
        class SSACheck(AnalysisPass):
            """
            Check SSA on variable `d`
            """
            _name = 'SSA_Check'

            def __init__(self):
                AnalysisPass.__init__(self)

            def run_pass(self, state):
                check(state.func_ir)
                return False

        class SSACheckPipeline(CompilerBase):
            """Inject SSACheck pass into the default pipeline following the SSA
            pass
            """

            def define_pipelines(self):
                pipeline = DefaultPassBuilder.define_nopython_pipeline(self.state, 'ssa_check_custom_pipeline')
                pipeline._finalized = False
                pipeline.add_pass_after(SSACheck, ReconstructSSA)
                pipeline.finalize()
                return [pipeline]

        @njit(pipeline_class=SSACheckPipeline)
        def py_func(a):
            c = a > 0
            if c:
                d = a + 5
            return c and d > 0
        py_func(10)