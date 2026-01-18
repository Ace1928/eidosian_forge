from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from .bases import (
from .primitive import Float, Int
from .singletons import Intrinsic, Undefined
class PositiveInt(Int):
    """
    Accept positive integers.

    .. deprecated:: 3.0.0

        Use ``Positive(Int)`` instead.
    """

    def __init__(self, default: Init[int]=Intrinsic, *, help: str | None=None) -> None:
        deprecated((3, 0, 0), 'Positive', 'Positive(Int)')
        super().__init__(default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if not 0 < value:
            raise ValueError(f'expected positive integer, got {value!r}')