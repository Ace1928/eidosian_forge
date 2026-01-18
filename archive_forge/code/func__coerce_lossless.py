from __future__ import annotations
import typing
from .expr import Expr, Var, Value, Unary, Binary, Cast
from ..types import CastKind, cast_kind
from .. import types
def _coerce_lossless(expr: Expr, type: types.Type) -> Expr:
    """Coerce ``expr`` to ``type`` by inserting a suitable :class:`Cast` node, if the cast is
    lossless.  Otherwise, raise a ``TypeError``."""
    kind = cast_kind(expr.type, type)
    if kind is CastKind.EQUAL:
        return expr
    if kind is CastKind.IMPLICIT:
        return Cast(expr, type, implicit=True)
    if kind is CastKind.LOSSLESS:
        return Cast(expr, type, implicit=False)
    if kind is CastKind.DANGEROUS:
        raise TypeError(f"cannot cast '{expr}' to '{type}' without loss of precision")
    raise TypeError(f"no cast is defined to take '{expr}' to '{type}'")