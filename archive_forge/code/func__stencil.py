import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def _stencil(mode, options):
    if mode != 'constant':
        raise ValueError('Unsupported mode style ' + mode)

    def decorated(func):
        from numba.core import compiler
        kernel_ir = compiler.run_frontend(func)
        return StencilFunc(kernel_ir, mode, options)
    return decorated