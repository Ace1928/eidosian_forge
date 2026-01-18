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
def _field_list_from_namedtuple(cls: TypeForm[Any], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    field_list = []
    field_defaults = getattr(cls, '_field_defaults')
    for name, typ in _resolver.get_type_hints(cls, include_extras=True).items():
        default = field_defaults.get(name, MISSING_NONPROP)
        if hasattr(default_instance, name):
            default = getattr(default_instance, name)
        if default_instance is MISSING_PROP:
            default = MISSING_PROP
        field_list.append(FieldDefinition.make(name=name, type_or_callable=typ, default=default, is_default_from_default_instance=True, helptext=_docstrings.get_field_docstring(cls, name)))
    return field_list