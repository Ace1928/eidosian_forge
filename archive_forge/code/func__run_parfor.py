import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
@classmethod
def _run_parfor(cls, test_func, args, swap_map=None):
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    test_ir = compiler.run_frontend(test_func)
    options = cpu.ParallelOptions(True)
    tp = MyPipeline(typingctx, targetctx, args, test_ir)
    typingctx.refresh()
    targetctx.refresh()
    inline_pass = inline_closurecall.InlineClosureCallPass(tp.state.func_ir, options, typed=True)
    inline_pass.run()
    rewrites.rewrite_registry.apply('before-inference', tp.state)
    untyped_passes.ReconstructSSA().run_pass(tp.state)
    tp.state.typemap, tp.state.return_type, tp.state.calltypes, _ = typed_passes.type_inference_stage(tp.state.typingctx, tp.state.targetctx, tp.state.func_ir, tp.state.args, None)
    typed_passes.PreLowerStripPhis().run_pass(tp.state)
    diagnostics = numba.parfors.parfor.ParforDiagnostics()
    preparfor_pass = numba.parfors.parfor.PreParforPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.typingctx, tp.state.targetctx, options, swapped=diagnostics.replaced_fns, replace_functions_map=swap_map)
    preparfor_pass.run()
    rewrites.rewrite_registry.apply('after-inference', tp.state)
    return (tp, options, diagnostics, preparfor_pass)