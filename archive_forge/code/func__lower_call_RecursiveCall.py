from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def _lower_call_RecursiveCall(self, fnty, expr, signature):
    argvals = self.fold_call_args(fnty, signature, expr.args, expr.vararg, expr.kws)
    rec_ov = fnty.get_overloads(signature.args)
    mangler = self.context.mangler or default_mangler
    abi_tags = self.fndesc.abi_tags
    mangled_name = mangler(rec_ov.qualname, signature.args, abi_tags=abi_tags, uid=rec_ov.uid)
    if self.builder.function.name.startswith(mangled_name):
        res = self.context.call_internal(self.builder, self.fndesc, signature, argvals)
    else:
        res = self.context.call_unresolved(self.builder, mangled_name, signature, argvals)
    return res