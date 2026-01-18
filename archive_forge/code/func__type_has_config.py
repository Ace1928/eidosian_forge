from __future__ import annotations as _annotations
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Set, TypeVar, Union, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import Literal, get_args, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel
from ._internal import _config, _generate_schema, _typing_extra
from .config import ConfigDict
from .json_schema import (
from .plugin._schema_validator import create_schema_validator
def _type_has_config(type_: Any) -> bool:
    """Returns whether the type has config."""
    try:
        return issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_)
    except TypeError:
        return False