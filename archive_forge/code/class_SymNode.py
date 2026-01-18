import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """

    def __init__(self, expr, shape_env, pytype, hint: Optional[Union[int, float, bool]], constant=None, fx_node=None):
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        if hint is not None:
            assert type(hint) is pytype or type(hint) is _to_symtype(pytype), f'Cannot create SymNode of type {pytype} with incompatible hint of type {type(hint)}'
        self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant
        self.fx_node = fx_node if self.shape_env._translation_validation_enabled else None

    def with_shape_env(self, shape_env: 'ShapeEnv') -> 'SymNode':
        return SymNode(self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node)

    @property
    def expr(self):
        return self.shape_env.replace(self._expr)

    def _update_hint(self):
        r = self.shape_env._maybe_evaluate_static(self.expr, compute_hint=True)
        if r is not None:
            self._hint = self.pytype(r) if not isinstance(r, SymTypes) else r

    @property
    def hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint

    def has_hint(self):
        if self._hint is None:
            self._update_hint()
        return self._hint is not None

    def require_hint(self, fallback=None):
        if self._hint is None:
            self._update_hint()
        if self._hint is None:
            if fallback is not None:
                return fallback
            return self.shape_env.size_hint(self.expr)
        return self._hint

    def maybe_as_int(self):
        if self.expr.is_number:
            return int(self.expr)
        else:
            return None

    def is_int(self):
        return self.pytype is int

    def is_float(self):
        return self.pytype is float

    def is_bool(self):
        return self.pytype is bool

    def wrap_int(self, num):
        assert type(num) is int
        import sympy
        return SymNode(sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num)

    def wrap_float(self, num):
        assert type(num) is float
        import sympy
        return SymNode(sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num)

    def wrap_bool(self, num):
        assert type(num) is bool
        import sympy
        return SymNode(sympy.true if num else sympy.false, self.shape_env, bool, num, constant=num, fx_node=num)

    def clone(self):
        return self

    def str(self):
        return f'{self.expr}'

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def abs(self) -> 'SymNode':
        return self._abs()

    def add(self, other) -> 'SymNode':
        return self._add(other)

    def sub(self, other) -> 'SymNode':
        return self._sub(other)

    def mul(self, other) -> 'SymNode':
        return self._mul(other)

    def mod(self, other) -> 'SymNode':
        return self._mod(other)

    def pow(self, other) -> 'SymNode':
        return self._pow(other)

    def and_(self, other) -> 'SymNode':
        return self._and_(other)

    def or_(self, other) -> 'SymNode':
        return self._or_(other)

    def truediv(self, other) -> 'SymNode':
        return self._truediv(other)

    def floordiv(self, other) -> 'SymNode':
        return self._floordiv(other)

    def lshift(self, other) -> 'SymNode':
        return self._lshift(other)

    def rshift(self, other) -> 'SymNode':
        return self._rshift(other)

    def sym_not(self) -> 'SymNode':
        return self._sym_not()

    def eq(self, other) -> 'SymNode':
        return self._eq(other)

    def ne(self, other) -> 'SymNode':
        return self._ne(other)

    def gt(self, other) -> 'SymNode':
        return self._gt(other)

    def lt(self, other) -> 'SymNode':
        return self._lt(other)

    def le(self, other) -> 'SymNode':
        return self._le(other)

    def ge(self, other) -> 'SymNode':
        return self._ge(other)

    def floor(self) -> 'SymNode':
        return self._floor()

    def sym_float(self) -> 'SymNode':
        return self._sym_float()

    def sym_int(self) -> 'SymNode':
        return self._sym_int()

    def ceil(self) -> 'SymNode':
        return self._ceil()

    def neg(self) -> 'SymNode':
        return self._neg()

    def sym_min(self, other) -> 'SymNode':
        return self._sym_min(other)

    def sym_max(self, other) -> 'SymNode':
        return self._sym_max(other)

    def sym_ite(self, then_val, else_val) -> 'SymNode':
        return self._sym_ite(then_val, else_val)

    def sym_sqrt(self) -> 'SymNode':
        return self._sym_sqrt()

    def is_contiguous(self, sizes, strides) -> 'SymNode':
        return self._is_contiguous(sizes, strides)

    def is_channels_last_contiguous_2d(self, sizes, strides) -> 'SymNode':
        return self._is_channels_last_contiguous_2d(sizes, strides)

    def is_channels_last_contiguous_3d(self, sizes, strides) -> 'SymNode':
        return self._is_channels_last_contiguous_3d(sizes, strides)

    def is_channels_last_strides_2d(self, sizes, strides) -> 'SymNode':
        return self._is_channels_last_strides_2d(sizes, strides)

    def is_channels_last_strides_3d(self, sizes, strides) -> 'SymNode':
        return self._is_channels_last_strides_3d(sizes, strides)

    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> 'SymNode':
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)

    def sym_or(self, other):
        return self.or_(other)

    def sym_and(self, other):
        return self.and_(other)

    def is_non_overlapping_and_dense(self, sizes, strides):
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))

    def int_(self):
        return self.guard_int('', 0)

    def guard_int(self, file, line):
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return int(r)
        except Exception:
            log.warning('Failed to convert to int: %s', r)
            raise

    def guard_float(self, file, line):
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node, expect_rational=False)
        try:
            return float(r)
        except Exception:
            log.warning('Failed to convert to float: %s', r)
            raise

    def guard_bool(self, file, line):
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return bool(r)
        except Exception:
            log.warning('Failed to convert to bool: %s', r)
            raise

    def expect_true(self, file, line):
        if self.has_hint():
            return self.guard_bool(file, line)
        return self.shape_env.defer_runtime_assert(self.expr, f'{file}:{line}', fx_node=self.fx_node)

    def expect_size(self, file, line):
        from torch.fx.experimental.symbolic_shapes import _advise_is_size
        b = self.ge(self.wrap_int(0))
        r = b.expect_true(file, line)
        if r and (not self.has_hint()):
            _advise_is_size(SymInt(self))
        return r

    def bool_(self):
        return self.guard_bool('', 0)

    def is_symbolic(self):
        return True

    def singleton_int(self):
        return None

    def is_constant(self):
        return False