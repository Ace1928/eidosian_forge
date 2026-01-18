from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def _uint_cast(from_: Uint, to_: Uint, /) -> CastKind:
    if from_.width == to_.width:
        return CastKind.EQUAL
    if from_.width < to_.width:
        return CastKind.LOSSLESS
    return CastKind.DANGEROUS