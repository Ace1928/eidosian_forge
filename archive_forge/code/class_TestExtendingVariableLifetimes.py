import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
class TestExtendingVariableLifetimes(SerialMixin, TestCase):

    def test_lifetime_basic(self):

        def get_ir(extend_lifetimes):

            class IRPreservingCompiler(CompilerBase):

                def define_pipelines(self):
                    pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                    pm.add_pass_after(PreserveIR, IRLegalization)
                    pm.finalize()
                    return [pm]

            @njit(pipeline_class=IRPreservingCompiler)
            def foo():
                a = 10
                b = 20
                c = a + b
                d = c / c
                return d
            with override_config('EXTEND_VARIABLE_LIFETIMES', extend_lifetimes):
                foo()
                cres = foo.overloads[foo.signatures[0]]
                func_ir = cres.metadata['preserved_ir']
            return func_ir

        def check(func_ir, expect):
            self.assertEqual(len(func_ir.blocks), 1)
            blk = next(iter(func_ir.blocks.values()))
            for expect_class, got_stmt in zip(expect, blk.body):
                self.assertIsInstance(got_stmt, expect_class)
        del_after_use_ir = get_ir(False)
        expect = [*(ir.Assign,) * 3, ir.Del, ir.Del, ir.Assign, ir.Del, ir.Assign, ir.Del, ir.Return]
        check(del_after_use_ir, expect)
        del_at_block_end_ir = get_ir(True)
        expect = [*(ir.Assign,) * 4, ir.Assign, *(ir.Del,) * 4, ir.Return]
        check(del_at_block_end_ir, expect)

    def test_dbg_extend_lifetimes(self):

        def get_ir(**options):

            class IRPreservingCompiler(CompilerBase):

                def define_pipelines(self):
                    pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                    pm.add_pass_after(PreserveIR, IRLegalization)
                    pm.finalize()
                    return [pm]

            @njit(pipeline_class=IRPreservingCompiler, **options)
            def foo():
                a = 10
                b = 20
                c = a + b
                d = c / c
                return d
            foo()
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return func_ir
        ir_debug = get_ir(debug=True)
        ir_debug_ext = get_ir(debug=True, _dbg_extend_lifetimes=True)
        ir_debug_no_ext = get_ir(debug=True, _dbg_extend_lifetimes=False)

        def is_del_grouped_at_the_end(fir):
            [blk] = fir.blocks.values()
            inst_is_del = [isinstance(stmt, ir.Del) for stmt in blk.body]
            not_dels = list(takewhile(operator.not_, inst_is_del))
            begin = len(not_dels)
            all_dels = list(takewhile(operator.truth, inst_is_del[begin:]))
            end = begin + len(all_dels)
            return end == len(inst_is_del) - 1
        self.assertTrue(is_del_grouped_at_the_end(ir_debug))
        self.assertTrue(is_del_grouped_at_the_end(ir_debug_ext))
        self.assertFalse(is_del_grouped_at_the_end(ir_debug_no_ext))