from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def _get_contained_special_types_from_type(cls: Type[Any], _parent_contained_dataclasses: Optional[Set[Type[Any]]]=None) -> Set[Type[Any]]:
    """Takes a dataclass type, and recursively searches its fields for dataclass or enum
    types."""
    assert _resolver.is_dataclass(cls)
    parent_contained_dataclasses = set() if _parent_contained_dataclasses is None else _parent_contained_dataclasses
    cls, _ = _resolver.unwrap_annotated(cls)
    cls, type_from_typevar = _resolver.resolve_generic_types(cls)
    contained_special_types = {cls}

    def handle_type(typ: Type[Any]) -> Set[Type[Any]]:
        if _resolver.is_dataclass(typ) and typ not in parent_contained_dataclasses:
            return _get_contained_special_types_from_type(typ, _parent_contained_dataclasses=contained_special_types | parent_contained_dataclasses)
        elif isinstance(typ, enum.EnumMeta):
            return {typ}
        return functools.reduce(set.union, map(handle_type, get_args(typ)), set())
    for typ in type_from_typevar.values():
        contained_special_types |= handle_type(typ)
    if cls in parent_contained_dataclasses:
        return contained_special_types
    for field in _resolver.resolved_fields(cls):
        contained_special_types |= handle_type(field.type)
    for subclass in cls.__subclasses__():
        contained_special_types |= handle_type(subclass)
    return contained_special_types