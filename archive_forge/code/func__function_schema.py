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
def _function_schema(self, schema: _core_utils.AnyFunctionSchema) -> JsonSchemaValue:
    if _core_utils.is_function_with_inner_schema(schema):
        return self.generate_inner(schema['schema'])
    return self.handle_invalid_for_json_schema(schema, f'core_schema.PlainValidatorFunctionSchema ({schema['function']})')