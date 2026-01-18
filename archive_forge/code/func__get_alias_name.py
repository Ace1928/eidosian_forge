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
def _get_alias_name(self, field: CoreSchemaField, name: str) -> str:
    if field['type'] == 'computed-field':
        alias: Any = field.get('alias', name)
    elif self.mode == 'validation':
        alias = field.get('validation_alias', name)
    else:
        alias = field.get('serialization_alias', name)
    if isinstance(alias, str):
        name = alias
    elif isinstance(alias, list):
        alias = cast('list[str] | str', alias)
        for path in alias:
            if isinstance(path, list) and len(path) == 1 and isinstance(path[0], str):
                name = path[0]
                break
    else:
        assert_never(alias)
    return name