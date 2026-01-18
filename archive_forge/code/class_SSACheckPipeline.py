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