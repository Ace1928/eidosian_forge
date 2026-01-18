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
def _lower_call_ExternalFunctionPointer(self, fnty, expr, signature):
    self.debug_print('# calling external function pointer')
    argvals = self.fold_call_args(fnty, signature, expr.args, expr.vararg, expr.kws)
    pointer = self.loadvar(expr.func.name)
    if fnty.requires_gil:
        self.init_pyapi()
        gil_state = self.pyapi.gil_ensure()
        newargvals = []
        pyvals = []
        for exptyp, gottyp, aval in zip(fnty.sig.args, signature.args, argvals):
            if exptyp == types.ffi_forced_object:
                self.incref(gottyp, aval)
                obj = self.pyapi.from_native_value(gottyp, aval, self.env_manager)
                newargvals.append(obj)
                pyvals.append(obj)
            else:
                newargvals.append(aval)
        res = self.context.call_function_pointer(self.builder, pointer, newargvals, fnty.cconv)
        for obj in pyvals:
            self.pyapi.decref(obj)
        self.pyapi.gil_release(gil_state)
    else:
        res = self.context.call_function_pointer(self.builder, pointer, argvals, fnty.cconv)
    return res