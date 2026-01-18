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
def _extract_discriminator(self, schema: core_schema.TaggedUnionSchema, one_of_choices: list[JsonDict]) -> str | None:
    """Extract a compatible OpenAPI discriminator from the schema and one_of choices that end up in the final
        schema."""
    openapi_discriminator: str | None = None
    if isinstance(schema['discriminator'], str):
        return schema['discriminator']
    if isinstance(schema['discriminator'], list):
        if len(schema['discriminator']) == 1 and isinstance(schema['discriminator'][0], str):
            return schema['discriminator'][0]
        for alias_path in schema['discriminator']:
            if not isinstance(alias_path, list):
                break
            if len(alias_path) != 1:
                continue
            alias = alias_path[0]
            if not isinstance(alias, str):
                continue
            alias_is_present_on_all_choices = True
            for choice in one_of_choices:
                while '$ref' in choice:
                    assert isinstance(choice['$ref'], str)
                    choice = self.get_schema_from_definitions(JsonRef(choice['$ref'])) or {}
                properties = choice.get('properties', {})
                if not isinstance(properties, dict) or alias not in properties:
                    alias_is_present_on_all_choices = False
                    break
            if alias_is_present_on_all_choices:
                openapi_discriminator = alias
                break
    return openapi_discriminator