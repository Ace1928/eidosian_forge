from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _get_schema_type_for_array(schema: Schema) -> Optional[Union[str, Tuple[str, ...]]]:
    from openapi_pydantic import Reference, Schema
    items = schema.items
    if isinstance(items, Schema):
        schema_type = APIProperty._cast_schema_list_type(items)
    elif isinstance(items, Reference):
        ref_name = items.ref.split('/')[-1]
        schema_type = ref_name
    else:
        raise ValueError(f'Unsupported array items: {items}')
    if isinstance(schema_type, str):
        schema_type = (schema_type,)
    return schema_type