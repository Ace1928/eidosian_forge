from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def _handle_other_schemas(self, schema: core_schema.CoreSchema, f: Walk) -> core_schema.CoreSchema:
    sub_schema = schema.get('schema', None)
    if sub_schema is not None:
        schema['schema'] = self.walk(sub_schema, f)
    return schema