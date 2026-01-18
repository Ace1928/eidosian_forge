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
def _try_field_list_from_sequence_inner(contained_type: TypeForm[Any], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    if default_instance in MISSING_SINGLETONS and (not is_nested_type(contained_type, MISSING_NONPROP)):
        return UnsupportedNestedTypeMessage(f'Sequence containing type {contained_type} should be parsed directly!')
    if isinstance(default_instance, Iterable) and all([not is_nested_type(type(x), x) for x in default_instance]):
        return UnsupportedNestedTypeMessage(f'Sequence with default {default_instance} should be parsed directly!')
    if default_instance in MISSING_SINGLETONS:
        raise _instantiators.UnsupportedTypeAnnotationError('For variable-length sequences over nested types, we need a default value to infer length from.')
    field_list = []
    for i, default_i in enumerate(default_instance):
        field_list.append(FieldDefinition.make(name=str(i), type_or_callable=contained_type, default=default_i, is_default_from_default_instance=True, helptext=''))
    return field_list