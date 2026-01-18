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
def _get_array_operator(self, ir_expr):
    ir_op = ir_expr.op
    if ir_op in ('unary', 'binop'):
        return ir_expr.fn
    elif ir_op == 'call':
        return self.typemap[ir_expr.func.name].typing_key
    raise NotImplementedError("Don't know how to find the operator for '{0}' expressions.".format(ir_op))