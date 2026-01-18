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
def _try_field_list_from_general_callable(f: Union[Callable, TypeForm[Any]], cls: Optional[TypeForm[Any]], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    if not callable(f):
        return UnsupportedNestedTypeMessage(f'Cannot extract annotations from {f}, which is not a callable type.')
    params = list(inspect.signature(f).parameters.values())
    if cls is not None:
        params = params[1:]
    out = _field_list_from_params(f, cls, params)
    if isinstance(out, UnsupportedNestedTypeMessage):
        return out
    if default_instance not in MISSING_SINGLETONS:
        for i, field in enumerate(out):
            out[i] = field.add_markers((_markers._OPTIONAL_GROUP,))
    return out