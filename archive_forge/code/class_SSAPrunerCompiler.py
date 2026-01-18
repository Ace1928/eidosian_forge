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
class SSAPrunerCompiler(CompilerBase):

    def define_pipelines(self):
        pm = PassManager('testing pm')
        pm.add_pass(TranslateByteCode, 'analyzing bytecode')
        pm.add_pass(IRProcessing, 'processing IR')
        pm.add_pass(ReconstructSSA, 'ssa')
        pm.add_pass(DeadBranchPrune, 'dead branch pruning')
        pm.add_pass(PreserveIR, 'preserves the IR as metadata')
        dpb = DefaultPassBuilder
        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)
        lowering_passes = dpb.define_nopython_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)
        pm.finalize()
        return [pm]