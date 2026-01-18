from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def _handle_ser_schemas(self, ser_schema: core_schema.SerSchema, f: Walk) -> core_schema.SerSchema:
    schema: core_schema.CoreSchema | None = ser_schema.get('schema', None)
    if schema is not None:
        ser_schema['schema'] = self.walk(schema, f)
    return_schema: core_schema.CoreSchema | None = ser_schema.get('return_schema', None)
    if return_schema is not None:
        ser_schema['return_schema'] = self.walk(return_schema, f)
    return ser_schema