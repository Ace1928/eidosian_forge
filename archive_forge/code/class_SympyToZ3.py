import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
class SympyToZ3:
    OPERATOR_HANDLES = {'add', 'mul', 'eq', 'ne', 'lt', 'gt', 'le', 'ge'}

    def __init__(self, validator: 'TranslationValidator') -> None:
        self._validator = validator
        self._ops = _Z3Ops(self._validator)

    def constant(self, value: Any, dtype: torch.dtype) -> z3.ExprRef:
        if dtype is torch.int64:
            return z3.IntVal(int(value))
        if dtype is torch.double:
            return z3.RealVal(float(value))
        if dtype is torch.bool:
            return z3.BoolVal(bool(value))
        raise ValueError(f'unsupported dtype (SympyToZ3): {dtype}')

    def truediv(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
        return self._ops.div(numerator, denominator)

    def floordiv(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
        return self._ops.floordiv(numerator, denominator)

    def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
        return self._ops.floordiv(numerator, denominator)

    def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
        return self._ops.pow(base, exp)

    def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
        return self._ops.mod(p, q)

    def __getattr__(self, name: str) -> Any:
        REPLACEMENT = {'and_': z3.And, 'or_': z3.Or, 'not_': z3.Not, 'floor': self._ops.floor, 'ceil': self._ops.ceil, 'minimum': self._ops.min, 'maximum': self._ops.max}
        if name in REPLACEMENT:
            return REPLACEMENT[name]
        if name in self.OPERATOR_HANDLES:
            return getattr(operator, name)
        raise AttributeError(f'unhandled operator: {name}')

    def run(self, expr: sympy.Basic) -> z3.ExprRef:
        return sympy_interp(self, self._validator.symbols, expr)