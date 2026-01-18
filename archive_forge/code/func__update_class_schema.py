from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
def _update_class_schema(self, json_schema: JsonSchemaValue, title: str | None, extra: Literal['allow', 'ignore', 'forbid'] | None, cls: type[Any], json_schema_extra: JsonDict | JsonSchemaExtraCallable | None) -> JsonSchemaValue:
    if '$ref' in json_schema:
        schema_to_update = self.get_schema_from_definitions(JsonRef(json_schema['$ref'])) or json_schema
    else:
        schema_to_update = json_schema
    if title is not None:
        schema_to_update.setdefault('title', title)
    if 'additionalProperties' not in schema_to_update:
        if extra == 'allow':
            schema_to_update['additionalProperties'] = True
        elif extra == 'forbid':
            schema_to_update['additionalProperties'] = False
    if isinstance(json_schema_extra, (staticmethod, classmethod)):
        json_schema_extra = json_schema_extra.__get__(cls)
    if isinstance(json_schema_extra, dict):
        schema_to_update.update(json_schema_extra)
    elif callable(json_schema_extra):
        if len(inspect.signature(json_schema_extra).parameters) > 1:
            json_schema_extra(schema_to_update, cls)
        else:
            json_schema_extra(schema_to_update)
    elif json_schema_extra is not None:
        raise ValueError(f"model_config['json_schema_extra']={json_schema_extra} should be a dict, callable, or None")
    return json_schema