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
def _field_list_from_dict(f: Union[Callable, TypeForm[Any]], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    if default_instance in MISSING_SINGLETONS or len(cast(dict, default_instance)) == 0:
        return UnsupportedNestedTypeMessage('Nested dictionary structures must have non-empty default instance specified.')
    field_list = []
    for k, v in cast(dict, default_instance).items():
        field_list.append(FieldDefinition.make(name=str(k) if not isinstance(k, enum.Enum) else k.name, type_or_callable=type(v), default=v, is_default_from_default_instance=True, helptext=None, call_argname_override=k))
    return field_list