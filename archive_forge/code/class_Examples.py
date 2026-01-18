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
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class Examples:
    """Add examples to a JSON schema.

    Examples should be a map of example names (strings)
    to example values (any valid JSON).

    If `mode` is set this will only apply to that schema generation mode,
    allowing you to add different examples for validation and serialization.
    """
    examples: dict[str, Any]
    mode: Literal['validation', 'serialization'] | None = None

    def __get_pydantic_json_schema__(self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        json_schema = handler(core_schema)
        if mode != handler.mode:
            return json_schema
        examples = json_schema.get('examples', {})
        examples.update(to_jsonable_python(self.examples))
        json_schema['examples'] = examples
        return json_schema

    def __hash__(self) -> int:
        return hash(type(self.mode))