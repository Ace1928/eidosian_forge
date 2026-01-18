from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
def _maybe_evaluate_binop(self, op, op_class, lhs, rhs, eval_in_python=('in', 'not in'), maybe_eval_in_python=('==', '!=', '<', '>', '<=', '>=')):
    res = op(lhs, rhs)
    if res.has_invalid_return_type:
        raise TypeError(f"unsupported operand type(s) for {res.op}: '{lhs.type}' and '{rhs.type}'")
    if self.engine != 'pytables' and (res.op in CMP_OPS_SYMS and getattr(lhs, 'is_datetime', False) or getattr(rhs, 'is_datetime', False)):
        return self._maybe_eval(res, self.binary_ops)
    if res.op in eval_in_python:
        return self._maybe_eval(res, eval_in_python)
    elif self.engine != 'pytables':
        if getattr(lhs, 'return_type', None) == object or getattr(rhs, 'return_type', None) == object:
            return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
    return res