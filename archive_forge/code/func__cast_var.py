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
def _cast_var(self, var, ty):
    """
        Cast a Numba IR variable to the given Numba type, returning a
        low-level value.
        """
    if isinstance(var, _VarArgItem):
        varty = self.typeof(var.vararg.name)[var.index]
        val = self.builder.extract_value(self.loadvar(var.vararg.name), var.index)
    else:
        varty = self.typeof(var.name)
        val = self.loadvar(var.name)
    return self.context.cast(self.builder, val, varty, ty)