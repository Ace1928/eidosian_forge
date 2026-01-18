import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
class TestBranchPrune(TestBranchPruneBase, SerialMixin):

    def test_single_if(self):

        def impl(x):
            if 1 == 0:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [True], None)

        def impl(x):
            if 1 == 1:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [False], None)

        def impl(x):
            if x is None:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [False], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

        def impl(x):
            if x == 10:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

        def impl(x):
            if x == 10:
                z = 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

        def impl(x):
            z = None
            y = z
            if x == y:
                return 100
        self.assert_prune(impl, (types.NoneType('none'),), [False], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

    def test_single_if_else(self):

        def impl(x):
            if x is None:
                return 3.14159
            else:
                return 1.61803
        self.assert_prune(impl, (types.NoneType('none'),), [False], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

    def test_single_if_const_val(self):

        def impl(x):
            if x == 100:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

        def impl(x):
            if 100 == x:
                return 3.14159
        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def test_single_if_else_two_const_val(self):

        def impl(x, y):
            if x == y:
                return 3.14159
            else:
                return 1.61803
        self.assert_prune(impl, (types.IntegerLiteral(100),) * 2, [None], 100, 100)
        self.assert_prune(impl, (types.NoneType('none'),) * 2, [False], None, None)
        self.assert_prune(impl, (types.IntegerLiteral(100), types.NoneType('none')), [True], 100, None)
        self.assert_prune(impl, (types.IntegerLiteral(100), types.IntegerLiteral(1000)), [None], 100, 1000)

    def test_single_if_else_w_following_undetermined(self):

        def impl(x):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                dead = 7
            if x_is_none_work:
                y = 10
            else:
                y = -3
            return y
        self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

        def impl(x):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                pass
            if x_is_none_work:
                y = 10
            else:
                y = -3
            return y
        if utils.PYVERSION >= (3, 10):
            self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
        else:
            self.assert_prune(impl, (types.NoneType('none'),), [None, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def test_double_if_else_rt_const(self):

        def impl(x):
            one_hundred = 100
            x_is_none_work = 4
            if x is None:
                x_is_none_work = 100
            else:
                dead = 7
            if x_is_none_work == one_hundred:
                y = 10
            else:
                y = -3
            return (y, x_is_none_work)
        self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def test_double_if_else_non_literal_const(self):

        def impl(x):
            one_hundred = 100
            if x == one_hundred:
                y = 3.14159
            else:
                y = 1.61803
            return y
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def test_single_two_branches_same_cond(self):

        def impl(x):
            if x is None:
                y = 10
            else:
                y = 40
            if x is not None:
                z = 100
            else:
                z = 400
            return (z, y)
        self.assert_prune(impl, (types.NoneType('none'),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_none(self):

        def impl(x=None):
            if x is None:
                y = 10
            else:
                y = 40
            if x is not None:
                z = 100
            else:
                z = 400
            return (z, y)
        self.assert_prune(impl, (types.Omitted(None),), [False, True], None)
        self.assert_prune(impl, (types.NoneType('none'),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_value(self):

        def impl(x=1000):
            if x == 1000:
                y = 10
            else:
                y = 40
            if x != 1000:
                z = 100
            else:
                z = 400
            return (z, y)
        self.assert_prune(impl, (types.Omitted(1000),), [None, None], 1000)
        self.assert_prune(impl, (types.IntegerLiteral(1000),), [None, None], 1000)
        self.assert_prune(impl, (types.IntegerLiteral(0),), [None, None], 0)
        self.assert_prune(impl, (types.NoneType('none'),), [True, False], None)

    def test_cond_rewrite_is_correct(self):

        def fn(x):
            if x is None:
                return 10
            return 12

        def check(func, arg_tys, bit_val):
            func_ir = compile_to_ir(func)
            before_branches = self.find_branches(func_ir)
            self.assertEqual(len(before_branches), 1)
            pred_var = before_branches[0].cond
            pred_defn = ir_utils.get_definition(func_ir, pred_var)
            self.assertEqual(pred_defn.op, 'call')
            condition_var = pred_defn.args[0]
            condition_op = ir_utils.get_definition(func_ir, condition_var)
            self.assertEqual(condition_op.op, 'binop')
            if self._DEBUG:
                print('=' * 80)
                print('before prune')
                func_ir.dump()
            dead_branch_prune(func_ir, arg_tys)
            if self._DEBUG:
                print('=' * 80)
                print('after prune')
                func_ir.dump()
            new_condition_defn = ir_utils.get_definition(func_ir, condition_var)
            self.assertTrue(isinstance(new_condition_defn, ir.Const))
            self.assertEqual(new_condition_defn.value, bit_val)
        check(fn, (types.NoneType('none'),), 1)
        check(fn, (types.IntegerLiteral(10),), 0)

    def test_global_bake_in(self):

        def impl(x):
            if _GLOBAL == 123:
                return x
            else:
                return x + 10
        self.assert_prune(impl, (types.IntegerLiteral(1),), [False], 1)
        global _GLOBAL
        tmp = _GLOBAL
        try:
            _GLOBAL = 5

            def impl(x):
                if _GLOBAL == 123:
                    return x
                else:
                    return x + 10
            self.assert_prune(impl, (types.IntegerLiteral(1),), [True], 1)
        finally:
            _GLOBAL = tmp

    def test_freevar_bake_in(self):
        _FREEVAR = 123

        def impl(x):
            if _FREEVAR == 123:
                return x
            else:
                return x + 10
        self.assert_prune(impl, (types.IntegerLiteral(1),), [False], 1)
        _FREEVAR = 12

        def impl(x):
            if _FREEVAR == 123:
                return x
            else:
                return x + 10
        self.assert_prune(impl, (types.IntegerLiteral(1),), [True], 1)

    def test_redefined_variables_are_not_considered_in_prune(self):

        def impl(array, a=None):
            if a is None:
                a = 0
            if a < 0:
                return 10
            return 30
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.NoneType('none')), [None, None], np.zeros((2, 3)), None)

    def test_comparison_operators(self):

        def impl(array, a=None):
            x = 0
            if a is None:
                return 10
            if a < 0:
                return 20
            return x
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.NoneType('none')), [False, 'both'], np.zeros((2, 3)), None)
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.float64), [None, None], np.zeros((2, 3)), 12.0)

    def test_redefinition_analysis_same_block(self):

        def impl(array, x, a=None):
            b = 2
            if x < 4:
                b = 12
            if a is None:
                a = 7
            else:
                b = 15
            if a < 0:
                return 10
            return 30 + b + a
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.float64, types.NoneType('none')), [None, False, None], np.zeros((2, 3)), 1.0, None)

    def test_redefinition_analysis_different_block_can_exec(self):

        def impl(array, x, a=None):
            b = 0
            if x > 5:
                a = 11
            if x < 4:
                b = 12
            if a is None:
                b += 5
            else:
                b += 7
                if a < 0:
                    return 10
            return 30 + b
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.float64, types.NoneType('none')), [None, None, None, None], np.zeros((2, 3)), 1.0, None)

    def test_redefinition_analysis_different_block_cannot_exec(self):

        def impl(array, x=None, a=None):
            b = 0
            if x is not None:
                a = 11
            if a is None:
                b += 5
            else:
                b += 7
            return 30 + b
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.NoneType('none'), types.NoneType('none')), [True, None], np.zeros((2, 3)), None, None)
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.NoneType('none'), types.float64), [True, None], np.zeros((2, 3)), None, 1.2)
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'), types.float64, types.NoneType('none')), [None, None], np.zeros((2, 3)), 1.2, None)

    def test_closure_and_nonlocal_can_prune(self):

        def impl():
            x = 1000

            def closure():
                nonlocal x
                x = 0
            closure()
            if x == 0:
                return True
            else:
                return False
        self.assert_prune(impl, (), [False])

    def test_closure_and_nonlocal_cannot_prune(self):

        def impl(n):
            x = 1000

            def closure(t):
                nonlocal x
                x = t
            closure(n)
            if x == 0:
                return True
            else:
                return False
        self.assert_prune(impl, (types.int64,), [None], 1)