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
def _get_dataclass_field_default(field: dataclasses.Field, parent_default_instance: Any) -> Tuple[Any, bool]:
    """Helper for getting the default instance for a dataclass field."""
    if parent_default_instance is MISSING_PROP:
        return (MISSING_PROP, False)
    if parent_default_instance not in MISSING_SINGLETONS and parent_default_instance is not None:
        if hasattr(parent_default_instance, field.name):
            return (getattr(parent_default_instance, field.name), True)
        else:
            warnings.warn(f'Could not find field {field.name} in default instance {parent_default_instance}, which has type {type(parent_default_instance)},', stacklevel=2)
    if field.default not in MISSING_SINGLETONS:
        default = field.default
        if type(default) is not type and dataclasses.is_dataclass(default):
            _ensure_dataclass_instance_used_as_default_is_frozen(field, default)
        return (default, False)
    if field.default_factory is not dataclasses.MISSING and (not (dataclasses.is_dataclass(field.type) and field.default_factory is field.type)):
        return (field.default_factory(), False)
    return (MISSING_NONPROP, False)