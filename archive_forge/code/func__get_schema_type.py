from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _get_schema_type(parameter: Parameter, schema: Optional[Schema]) -> SCHEMA_TYPE:
    if schema is None:
        return None
    schema_type: SCHEMA_TYPE = APIProperty._cast_schema_list_type(schema)
    if schema_type == 'array':
        schema_type = APIProperty._get_schema_type_for_array(schema)
    elif schema_type == 'object':
        raise NotImplementedError('Objects not yet supported')
    elif schema_type in PRIMITIVE_TYPES:
        if schema.enum:
            schema_type = APIProperty._get_schema_type_for_enum(parameter, schema)
        else:
            pass
    else:
        raise NotImplementedError(f'Unsupported type: {schema_type}')
    return schema_type