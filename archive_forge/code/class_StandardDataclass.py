from __future__ import annotations as _annotations
import dataclasses
import typing
import warnings
from functools import partial, wraps
from typing import Any, Callable, ClassVar
from pydantic_core import (
from typing_extensions import TypeGuard
from ..errors import PydanticUndefinedAnnotation
from ..fields import FieldInfo
from ..plugin._schema_validator import create_schema_validator
from ..warnings import PydanticDeprecatedSince20
from . import _config, _decorators, _typing_extra
from ._fields import collect_dataclass_fields
from ._generate_schema import GenerateSchema
from ._generics import get_standard_typevars_map
from ._mock_val_ser import set_dataclass_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
class StandardDataclass(typing.Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]
    __dataclass_params__: ClassVar[Any]
    __post_init__: ClassVar[Callable[..., None]]

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass