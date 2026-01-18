from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@classmethod
def _process_object_schema(cls, schema: Schema, spec: OpenAPISpec, references_used: List[str]) -> Tuple[Union[str, List[str], None], List['APIRequestBodyProperty']]:
    from openapi_pydantic import Reference
    properties = []
    required_props = schema.required or []
    if schema.properties is None:
        raise ValueError(f'No properties found when processing object schema: {schema}')
    for prop_name, prop_schema in schema.properties.items():
        if isinstance(prop_schema, Reference):
            ref_name = prop_schema.ref.split('/')[-1]
            if ref_name not in references_used:
                references_used.append(ref_name)
                prop_schema = spec.get_referenced_schema(prop_schema)
            else:
                continue
        properties.append(cls.from_schema(schema=prop_schema, name=prop_name, required=prop_name in required_props, spec=spec, references_used=references_used))
    return (schema.type, properties)