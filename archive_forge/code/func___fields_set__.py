from __future__ import annotations as _annotations
import operator
import sys
import types
import typing
import warnings
from copy import copy, deepcopy
from typing import Any, ClassVar
import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .warnings import PydanticDeprecatedSince20
@property
@typing_extensions.deprecated('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=None)
def __fields_set__(self) -> set[str]:
    warnings.warn('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=PydanticDeprecatedSince20)
    return self.__pydantic_fields_set__