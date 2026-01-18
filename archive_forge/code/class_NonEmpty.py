from __future__ import annotations
import logging # isort:skip
from collections.abc import (
from typing import TYPE_CHECKING, Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .descriptors import ColumnDataPropertyDescriptor
from .enum import Enum
from .numeric import Int
from .singletons import Intrinsic, Undefined
from .wrappers import (
class NonEmpty(SingleParameterizedProperty[TSeq]):
    """ Allows only non-empty containers. """

    def __init__(self, type_param: TypeOrInst[TSeq], *, default: Init[TSeq]=Intrinsic, help: str | None=None) -> None:
        super().__init__(type_param, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        if not value:
            msg = '' if not detail else 'Expected a non-empty container'
            raise ValueError(msg)