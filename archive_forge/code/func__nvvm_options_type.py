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
def _nvvm_options_type(x):
    if x is None:
        return None
    else:
        assert isinstance(x, dict)
        return x