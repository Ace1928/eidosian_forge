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
class CUDACompiler(CompilerBase):

    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('cuda')
        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)
        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)
        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)
        pm.finalize()
        return [pm]

    def define_cuda_lowering_pipeline(self, state):
        pm = PassManager('cuda_lowering')
        pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
        pm.add_pass(AnnotateTypes, 'annotate types')
        pm.add_pass(CreateLibrary, 'create library')
        pm.add_pass(NativeLowering, 'native lowering')
        pm.add_pass(CUDABackend, 'cuda backend')
        pm.finalize()
        return pm