from __future__ import annotations
from enum import IntEnum
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, TypeVar
import attrs
from attrs import define, field
from fontTools.ufoLib import UFOReader
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.misc import AttrDictMixin
from ufoLib2.serde import serde
from .woff import (
def _converter_setter_property(cls: type[Any], converter: Callable[[Any], Any], name: str | None=None) -> Any:
    if name is None:
        class_name = cls.__name__
        name = f'_{class_name[0].lower()}{class_name[1:]}'
    attr_name: str = name

    def getter(self: Any) -> Any:
        return getattr(self, attr_name)

    def setter(self: Any, value: Any) -> None:
        setattr(self, attr_name, converter(value))
    return property(getter, setter)