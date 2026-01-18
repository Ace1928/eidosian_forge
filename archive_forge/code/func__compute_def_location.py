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
def _compute_def_location(self):
    defn_loc = self.func_ir.loc.with_lineno(self.func_ir.loc.line + 1)
    if self.context.enable_debuginfo:
        fn = self.func_ir.func_id.func
        optional_lno = get_func_body_first_lineno(fn)
        if optional_lno is not None:
            offset = optional_lno - 1
            defn_loc = self.func_ir.loc.with_lineno(offset)
        else:
            msg = f'Could not find source for function: {self.func_ir.func_id.func}. Debug line information may be inaccurate.'
            warnings.warn(NumbaDebugInfoWarning(msg))
    return defn_loc