from numba import jit, njit
from numba.core import types, ir, config, compiler
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.tests.support import IRPreservingTestPipeline
import numpy as np
import unittest
class TestCopyPropagate(unittest.TestCase):

    def test1(self):
        typingctx = cpu_target.typing_context
        targetctx = cpu_target.target_context
        test_ir = compiler.run_frontend(test_will_propagate)
        typingctx.refresh()
        targetctx.refresh()
        args = (types.int64, types.int64, types.int64)
        typemap, return_type, calltypes, _ = type_inference_stage(typingctx, targetctx, test_ir, args, None)
        type_annotation = type_annotations.TypeAnnotation(func_ir=test_ir, typemap=typemap, calltypes=calltypes, lifted=(), lifted_from=None, args=args, return_type=return_type, html_output=config.HTML)
        in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
        apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)
        self.assertFalse(findAssign(test_ir, 'x1'))

    def test2(self):
        typingctx = cpu_target.typing_context
        targetctx = cpu_target.target_context
        test_ir = compiler.run_frontend(test_wont_propagate)
        typingctx.refresh()
        targetctx.refresh()
        args = (types.int64, types.int64, types.int64)
        typemap, return_type, calltypes, _ = type_inference_stage(typingctx, targetctx, test_ir, args, None)
        type_annotation = type_annotations.TypeAnnotation(func_ir=test_ir, typemap=typemap, calltypes=calltypes, lifted=(), lifted_from=None, args=args, return_type=return_type, html_output=config.HTML)
        in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
        apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)
        self.assertTrue(findAssign(test_ir, 'x'))

    def test_input_ir_extra_copies(self):
        """make sure Interpreter._remove_unused_temporaries() has removed extra copies
        in the IR in simple cases so copy propagation is faster
        """

        def test_impl(a):
            b = a + 3
            return b
        j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
        self.assertEqual(test_impl(5), j_func(5))
        fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
        self.assertTrue(len(fir.blocks) == 1)
        block = next(iter(fir.blocks.values()))
        b_found = False
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and stmt.target.name == 'b':
                b_found = True
                self.assertTrue(isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop' and (stmt.value.lhs.name == 'a'))
        self.assertTrue(b_found)

    def test_input_ir_copy_remove_transform(self):
        """make sure Interpreter._remove_unused_temporaries() does not generate
        invalid code for rare chained assignment cases
        """

        def impl1(a):
            b = c = a + 1
            return (b, c)

        def impl2(A, i, a):
            b = A[i] = a + 1
            return (b, A[i] + 2)

        def impl3(A, a):
            b = A.a = a + 1
            return (b, A.a + 2)

        class C:
            pass
        self.assertEqual(impl1(5), njit(impl1)(5))
        self.assertEqual(impl2(np.ones(3), 0, 5), njit(impl2)(np.ones(3), 0, 5))
        self.assertEqual(impl3(C(), 5), jit(forceobj=True)(impl3)(C(), 5))