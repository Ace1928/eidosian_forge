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
def _named_required_fields_schema(self, named_required_fields: Sequence[tuple[str, bool, CoreSchemaField]]) -> JsonSchemaValue:
    properties: dict[str, JsonSchemaValue] = {}
    required_fields: list[str] = []
    for name, required, field in named_required_fields:
        if self.by_alias:
            name = self._get_alias_name(field, name)
        try:
            field_json_schema = self.generate_inner(field).copy()
        except PydanticOmit:
            continue
        if 'title' not in field_json_schema and self.field_title_should_be_set(field):
            title = self.get_title_from_name(name)
            field_json_schema['title'] = title
        field_json_schema = self.handle_ref_overrides(field_json_schema)
        properties[name] = field_json_schema
        if required:
            required_fields.append(name)
    json_schema = {'type': 'object', 'properties': properties}
    if required_fields:
        json_schema['required'] = required_fields
    return json_schema