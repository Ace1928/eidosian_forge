from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@staticmethod
def _cast_schema_list_type(schema: Schema) -> Optional[Union[str, Tuple[str, ...]]]:
    type_ = schema.type
    if not isinstance(type_, list):
        return type_
    else:
        return tuple(type_)