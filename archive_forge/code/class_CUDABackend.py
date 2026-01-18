from llvmlite import ir
from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, typing, funcdesc, config, compiler, sigutils
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass,
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.typed_passes import (IRLegalization, NativeLowering,
from warnings import warn
from numba.cuda.api import get_current_device
from numba.cuda.target import CUDACABICallConv
@register_pass(mutates_CFG=True, analysis_only=False)
class CUDABackend(LoweringPass):
    _name = 'cuda_backend'

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)
        state.cr = cuda_compile_result(typing_context=state.typingctx, target_context=state.targetctx, typing_error=state.status.fail_reason, type_annotation=state.type_annotation, library=state.library, call_helper=lowered.call_helper, signature=signature, fndesc=lowered.fndesc)
        return True