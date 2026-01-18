import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
class ValueRangeAnalysis(SymPyValueRangeAnalysis):

    def __init__(self):
        self.name = 'ValueRangeAnalysis'
        boolean_operators = ('xor', 'logical_and', 'logical_or', 'logical_not')
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args, **kwargs):
        return ValueRanges(sympy.false, sympy.true)

    @staticmethod
    def default_handler(*args, **kwargs):
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr):
        return ValueRanges.unknown()

    def store(self, name, index, value, mode=None):
        return

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        return ValueRanges.unknown()

    def index_expr(self, index, dtype):
        assert isinstance(index, ValueRanges)
        return index

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype]=None):
        x = ValueRanges.wrap(x)
        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x, dtype):
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                try:
                    return sympy.Integer(x)
                except TypeError:
                    return x
        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    def square(x):
        return ValueRanges.convex_min_zero_map(x, lambda y: y * y)

    @staticmethod
    def neg(x):
        return ValueRanges.decreasing_map(x, operator.neg)

    @classmethod
    def truncdiv(cls, a, b):
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        def trunc(x):
            return sympy.Integer(x) if x.is_finite else x
        return ValueRanges.increasing_map(x, trunc)

    @classmethod
    def sub(cls, a, b):
        return cls.add(a, cls.neg(b))

    def __getattr__(self, name):
        log.debug('unhandled ValueRange op %s', name)
        return self.default_handler