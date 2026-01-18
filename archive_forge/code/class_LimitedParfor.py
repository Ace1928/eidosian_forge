import numba
import numba.parfors.parfor
from numba import njit
from numba.core import ir_utils
from numba.core import types, ir,  compiler
from numba.core.registry import cpu_target
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.core.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
import numpy as np
from numba.tests.support import skip_parfors_unsupported, needs_blas
import unittest
@register_pass(analysis_only=False, mutates_CFG=True)
class LimitedParfor(FunctionPass):
    _name = 'limited_parfor'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        parfor_pass = numba.parfors.parfor.ParforPass(state.func_ir, state.typemap, state.calltypes, state.return_type, state.typingctx, state.flags.auto_parallel, state.flags, state.metadata, state.parfor_diagnostics)
        remove_dels(state.func_ir.blocks)
        parfor_pass.array_analysis.run(state.func_ir.blocks)
        parfor_pass._convert_loop(state.func_ir.blocks)
        remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir, state.typemap)
        numba.parfors.parfor.get_parfor_params(state.func_ir.blocks, parfor_pass.options.fusion, parfor_pass.nested_fusion_info)
        return True