from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def _binary_relation(op: Binary.Op, left: typing.Any, right: typing.Any) -> Expr:
    left, right = _lift_binary_operands(left, right)
    if left.type.kind is not right.type.kind or left.type.kind is types.Bool:
        raise TypeError(f"invalid types for '{op}': '{left.type}' and '{right.type}'")
    type = types.greater(left.type, right.type)
    return Binary(op, _coerce_lossless(left, type), _coerce_lossless(right, type), types.Bool())