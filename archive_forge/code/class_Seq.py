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
class Seq(ContainerProperty[T]):
    """ Accept non-string ordered sequences of values, e.g. list, tuple, array.

    """

    def __init__(self, item_type: TypeOrInst[Property[T]], *, default: Init[T]=Undefined, help: str | None=None) -> None:
        super().__init__(item_type, default=default, help=help)

    @property
    def item_type(self):
        return self.type_params[0]

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, True)
        if self._is_seq(value) and all((self.item_type.is_valid(item) for item in value)):
            return
        if self._is_seq(value):
            invalid = []
            for item in value:
                if not self.item_type.is_valid(item):
                    invalid.append(item)
            msg = '' if not detail else f'expected an element of {self}, got seq with invalid items {invalid!r}'
            raise ValueError(msg)
        msg = '' if not detail else f'expected an element of {self}, got {value!r}'
        raise ValueError(msg)

    @classmethod
    def _is_seq(cls, value: Any) -> bool:
        return (isinstance(value, Sequence) or cls._is_seq_like(value)) and (not isinstance(value, str))

    @classmethod
    def _is_seq_like(cls, value: Any) -> bool:
        return isinstance(value, (Container, Sized, Iterable)) and hasattr(value, '__getitem__') and (not isinstance(value, Mapping))