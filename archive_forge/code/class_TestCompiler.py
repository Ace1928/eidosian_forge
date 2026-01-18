from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
class TestCompiler(Compiler):

    def define_pipelines(self):
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        pm.add_pass_after(_InjectDelsPass, inject_after)
        pm.finalize()
        return [pm]