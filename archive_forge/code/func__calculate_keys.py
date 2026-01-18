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
@typing_extensions.deprecated('The private method `_calculate_keys` will be removed and should no longer be used.', category=None)
def _calculate_keys(self, *args: Any, **kwargs: Any) -> Any:
    warnings.warn('The private method `_calculate_keys` will be removed and should no longer be used.', category=PydanticDeprecatedSince20)
    from .deprecated import copy_internals
    return copy_internals._calculate_keys(self, *args, **kwargs)