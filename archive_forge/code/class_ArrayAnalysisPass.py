import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
@register_pass(analysis_only=False, mutates_CFG=True)
class ArrayAnalysisPass(FunctionPass):
    _name = 'array_analysis_pass'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.array_analysis = ArrayAnalysis(state.typingctx, state.func_ir, state.typemap, state.calltypes)
        state.array_analysis.run(state.func_ir.blocks)
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()
        state.func_ir_copies.append(state.func_ir.copy())
        if state.test_idempotence and len(state.func_ir_copies) > 1:
            state.test_idempotence(state.func_ir_copies)
        return False