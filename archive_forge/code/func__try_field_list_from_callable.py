from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def _try_field_list_from_callable(f: Union[Callable, TypeForm[Any]], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    f, found_subcommand_configs = _resolver.unwrap_annotated(f, conf._confstruct._SubcommandConfiguration)
    if len(found_subcommand_configs) > 0:
        default_instance = found_subcommand_configs[0].default
    f, type_from_typevar = _resolver.resolve_generic_types(f)
    f = _resolver.apply_type_from_typevar(f, type_from_typevar)
    f = _resolver.unwrap_newtype_and_narrow_subtypes(f, default_instance)
    f = _resolver.narrow_collection_types(f, default_instance)
    f_origin = _resolver.unwrap_origin_strip_extras(cast(TypeForm, f))
    cls: Optional[TypeForm[Any]] = None
    if inspect.isclass(f):
        cls = f
        if hasattr(cls, '__init__') and cls.__init__ is not object.__init__:
            f = cls.__init__
        elif hasattr(cls, '__new__') and cls.__new__ is not object.__new__:
            f = cls.__new__
        else:
            return UnsupportedNestedTypeMessage(f'Cannot instantiate class {cls} with no unique __init__ or __new__ method.')
        f_origin = cls
    if cls is not None:
        for match, field_list_from_class in ((is_typeddict, _field_list_from_typeddict), (_resolver.is_namedtuple, _field_list_from_namedtuple), (_resolver.is_dataclass, _field_list_from_dataclass), (_is_attrs, _field_list_from_attrs), (_is_pydantic, _field_list_from_pydantic)):
            if match(cls):
                return field_list_from_class(cls, default_instance)
    if f_origin is tuple or cls is tuple:
        return _field_list_from_tuple(f, default_instance)
    elif f_origin in (collections.abc.Mapping, dict) or cls in (collections.abc.Mapping, dict):
        return _field_list_from_dict(f, default_instance)
    elif f_origin in (list, set, typing.Sequence) or cls in (list, set, typing.Sequence):
        return _field_list_from_nontuple_sequence_checked(f, default_instance)
    if cls is not None and cls in _known_parsable_types or _resolver.unwrap_origin_strip_extras(f) in _known_parsable_types:
        return UnsupportedNestedTypeMessage(f'{f} should be parsed directly!')
    elif cls is not None and issubclass(_resolver.unwrap_origin_strip_extras(cls), os.PathLike) and _instantiators.is_type_string_converter(cls):
        return UnsupportedNestedTypeMessage(f'PathLike {cls} should be parsed directly!')
    else:
        return _try_field_list_from_general_callable(f, cls, default_instance)