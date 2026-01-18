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
@register_pass(mutates_CFG=False, analysis_only=False)
class CreateLibrary(LoweringPass):
    """
    Create a CUDACodeLibrary for the NativeLowering pass to populate. The
    NativeLowering pass will create a code library if none exists, but we need
    to set it up with nvvm_options from the flags if they are present.
    """
    _name = 'create_library'

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        codegen = state.targetctx.codegen()
        name = state.func_id.func_qualname
        nvvm_options = state.flags.nvvm_options
        state.library = codegen.create_library(name, nvvm_options=nvvm_options)
        state.library.enable_object_caching()
        return True