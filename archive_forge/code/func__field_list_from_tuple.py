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
def _field_list_from_tuple(f: Union[Callable, TypeForm[Any]], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    field_list: List[FieldDefinition] = []
    children = get_args(f)
    if Ellipsis in children:
        return _try_field_list_from_sequence_inner(next(iter(set(children) - {Ellipsis})), default_instance)
    if len(children) == 0:
        if default_instance in MISSING_SINGLETONS:
            return UnsupportedNestedTypeMessage('If contained types of a tuple are not specified in the annotation, a default instance must be specified.')
        else:
            assert isinstance(default_instance, tuple)
            children = tuple((type(x) for x in default_instance))
    if default_instance in MISSING_SINGLETONS or default_instance is EXCLUDE_FROM_CALL:
        default_instance = (default_instance,) * len(children)
    for i, child in enumerate(children):
        default_i = default_instance[i]
        field_list.append(FieldDefinition.make(name=str(i), type_or_callable=child, default=default_i, is_default_from_default_instance=True, helptext=''))
    contains_nested = False
    for field in field_list:
        if get_origin(field.type_or_callable) is Union:
            for option in get_args(field.type_or_callable):
                contains_nested |= is_nested_type(option, MISSING_NONPROP)
        contains_nested |= is_nested_type(field.type_or_callable, field.default)
        if contains_nested:
            break
    if not contains_nested:
        return UnsupportedNestedTypeMessage('Tuple does not contain any nested structures.')
    return field_list