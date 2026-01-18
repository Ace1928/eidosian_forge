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
def _translate_expr(self, ir_expr):
    """Translate the given expression from Numba IR to an array expression
        tree.
        """
    ir_op = ir_expr.op
    if ir_op == 'arrayexpr':
        return ir_expr.expr
    operands_or_args = [self.const_assigns.get(op_var.name, op_var) for op_var in self._get_operands(ir_expr)]
    return (self._get_array_operator(ir_expr), operands_or_args)