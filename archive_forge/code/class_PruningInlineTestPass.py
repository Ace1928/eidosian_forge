import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
@register_pass(analysis_only=False, mutates_CFG=True)
class PruningInlineTestPass(FunctionPass):
    _name = 'pruning_inline_test_pass'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        assert len(state.func_ir.blocks) == 1
        block = list(state.func_ir.blocks.values())[0]
        for i, stmt in enumerate(block.body):
            if guard(find_callname, state.func_ir, stmt.value) is not None:
                inline_closure_call(state.func_ir, {}, block, i, foo.py_func, state.typingctx, state.targetctx, (state.typemap[stmt.value.args[0].name],), state.typemap, state.calltypes)
                break
        return True