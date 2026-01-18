from numba.core.compiler import Compiler, DefaultPassBuilder
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core.untyped_passes import InlineInlinables
from numba.core.typed_passes import IRLegalization
from numba import jit, objmode, njit, cfunc
from numba.core import types, postproc, errors
from numba.core.ir import FunctionIR
from numba.tests.support import TestCase
@register_pass(mutates_CFG=False, analysis_only=False)
class _InjectDelsPass(base):
    """
            This pass injects ir.Del nodes into the IR
            """
    _name = 'inject_dels_%s' % str(base)

    def __init__(self):
        base.__init__(self)

    def run_pass(self, state):
        pp = postproc.PostProcessor(state.func_ir)
        pp.run(emit_dels=True)
        return True