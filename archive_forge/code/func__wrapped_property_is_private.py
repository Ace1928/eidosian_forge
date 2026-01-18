from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
def _wrapped_property_is_private(property_: cached_property | property) -> bool:
    """Returns true if provided property is private, False otherwise."""
    wrapped_name: str = ''
    if isinstance(property_, property):
        wrapped_name = getattr(property_.fget, '__name__', '')
    elif isinstance(property_, cached_property):
        wrapped_name = getattr(property_.func, '__name__', '')
    return wrapped_name.startswith('_') and (not wrapped_name.startswith('__'))