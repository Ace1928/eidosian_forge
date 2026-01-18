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
class SymPyValueRangeAnalysis:
    """
    It gives bounds on a SymPy operator given bounds on its arguments
    See the function `bound_sympy` for a function that applies this logic to a full SymPy expression
    """

    @staticmethod
    def constant(value, dtype):
        is_python = isinstance(value, (int, float, bool))
        assert is_python or isinstance(value, (BooleanAtom, sympy.Integer, sympy.Number))
        if isinstance(value, SupportsFloat) and math.isnan(value):
            return ValueRanges.unknown()
        if is_python:
            type_ = dtype_to_type(dtype)
            value = type_(value)
        elif dtype == torch.bool:
            assert isinstance(value, BooleanAtom)
        elif dtype.is_floating_point:
            assert not value.is_finite or value.is_real
        else:
            assert value.is_integer
        return ValueRanges.wrap(value)

    @staticmethod
    def not_(a):
        a = ValueRanges.wrap(a)
        assert a.is_bool
        return ValueRanges.decreasing_map(a, sympy.Not)

    @staticmethod
    def or_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.Or)

    @staticmethod
    def and_(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.And)

    @staticmethod
    def eq(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and (a.lower == b.lower):
            return ValueRanges.wrap(sympy.true)
        elif a.lower > b.upper or b.lower > a.upper:
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @classmethod
    def lt(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(cls.not_(a), b)
        else:
            if a.upper < b.lower:
                return ValueRanges.wrap(sympy.true)
            elif a.lower >= b.upper:
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        return cls.lt(b, a)

    @classmethod
    def le(cls, a, b):
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        return cls.not_(cls.lt(a, b))

    @staticmethod
    def add(a, b):
        return ValueRanges.coordinatewise_increasing_map(a, b, operator.add)

    @classmethod
    def mul(cls, a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(a, b)

        def safe_mul(a, b):
            if a == 0:
                return a
            elif b == 0:
                return b
            else:
                return a * b
        return ValueRanges.coordinatewise_monotone_map(a, b, safe_mul)

    @classmethod
    def div(cls, a, b):
        return cls.truediv(a, b)

    @staticmethod
    def truediv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.truediv)

    @staticmethod
    def floordiv(a, b):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.floordiv)

    @staticmethod
    def mod(x, y):
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.is_singleton() and y.is_singleton() and (y.lower != 0):
            return ValueRanges.wrap(x.lower % y.lower)
        if y.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges(0, y.upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def is_non_overlapping_and_dense_indicator(cls, *args):
        return ValueRanges.unknown()

    @classmethod
    def pow(cls, a, b):

        def is_integer(val):
            return isinstance(val, int) or (hasattr(val, 'is_integer') and val.is_integer)
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if not b.is_singleton():
            return ValueRanges.unknown()
        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a ** b
            if not r.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)
        if b == 0:
            if not a.lower.is_finite:
                return ValueRanges.unknown()
            type_ = sympy.Float if a.lower.is_real else sympy.Integer
            return ValueRanges.wrap(type_(1))
        if b < 0:
            a = cls.reciprocal(a)
            b = -b
        if a == ValueRanges.unknown():
            return ValueRanges.unknown()
        if not is_integer(b):
            if a.lower >= 0:
                return ValueRanges.increasing_map(a, lambda x: x ** b)
            else:
                return ValueRanges.unknown()
        elif b % 2 == 0:
            return ValueRanges.convex_min_zero_map(a, lambda x: x ** b)
        else:
            return ValueRanges.increasing_map(a, lambda x: x ** b)

    @staticmethod
    def reciprocal(x):
        """ Needed as it's used in pow, but it won't appear on a SymPy expression """
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: 1 / y)

    @staticmethod
    def abs(x):
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.exponential.exp)

    @staticmethod
    def log(x):
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.log)

    @classmethod
    def minimum(cls, a, b):
        return cls.min_or_max(a, b, sympy.Min)

    @classmethod
    def maximum(cls, a, b):
        return cls.min_or_max(a, b, sympy.Max)

    @staticmethod
    def min_or_max(a, b, fn):
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        def fn_(x, y):
            if x.is_Integer and y.is_Integer:
                result_type = sympy.Integer
            elif x.is_rational and y.is_rational:
                result_type = sympy.Rational
            else:
                assert x.is_real or not x.is_finite or y.is_real or (not y.is_finite)
                result_type = sympy.Float
            return fn(result_type(x), result_type(y))
        return ValueRanges.coordinatewise_increasing_map(a, b, fn_)

    @classmethod
    def floor(cls, x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.floor)

    @classmethod
    def ceil(cls, x):
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.ceiling)

    @staticmethod
    def sqrt(x):
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @staticmethod
    def where(a, b, c):
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        assert a.is_bool
        assert b.is_bool == c.is_bool
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(sympy.Min(b.lower, c.lower), sympy.Max(b.upper, c.upper))

    @staticmethod
    def expr_cond_pair(a, b):
        assert b.is_bool, f"expect cond_expr's ValueRange to be a boolean range but got {b}"
        return (a, b)

    @staticmethod
    def piecewise(*ranges):
        init_range = None
        for expr_range, cond_range in ranges:
            if sympy.true in cond_range:
                if init_range is None:
                    init_range = expr_range
                else:
                    init_range = init_range | expr_range
        return init_range