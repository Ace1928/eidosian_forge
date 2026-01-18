import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
def _match_array_expr(self, instr, expr, target_name):
    """
        Find whether the given assignment (*instr*) of an expression (*expr*)
        to variable *target_name* is an array expression.
        """
    expr_op = expr.op
    array_assigns = self.array_assigns
    if expr_op in ('unary', 'binop') and expr.fn in npydecl.supported_array_operators:
        if all((self.typemap[var.name].is_internal for var in expr.list_vars())):
            array_assigns[target_name] = instr
    elif expr_op == 'call' and expr.func.name in self.typemap:
        func_type = self.typemap[expr.func.name]
        if isinstance(func_type, types.Function):
            func_key = func_type.typing_key
            if _is_ufunc(func_key):
                if not self._has_explicit_output(expr, func_key):
                    array_assigns[target_name] = instr