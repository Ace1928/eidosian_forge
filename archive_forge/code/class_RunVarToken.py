from __future__ import annotations
from typing import Generic, TypeVar, cast
import attrs
from .._util import NoPublicConstructor, final
from . import _run
@final
@attrs.define(eq=False, hash=False)
class RunVarToken(Generic[T], metaclass=NoPublicConstructor):
    _var: RunVar[T]
    previous_value: T | type[_NoValue] = _NoValue
    redeemed: bool = attrs.field(default=False, init=False)

    @classmethod
    def _empty(cls, var: RunVar[T]) -> RunVarToken[T]:
        return cls._create(var)