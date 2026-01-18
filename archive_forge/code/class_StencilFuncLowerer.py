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
class StencilFuncLowerer(object):
    """Callable class responsible for lowering calls to a specific StencilFunc.
    """

    def __init__(self, sf):
        self.stencilFunc = sf

    def __call__(self, context, builder, sig, args):
        cres = self.stencilFunc.compile_for_argtys(sig.args, {}, sig.return_type, None)
        res = context.call_internal(builder, cres.fndesc, sig, args)
        context.add_linking_libs([cres.library])
        return res