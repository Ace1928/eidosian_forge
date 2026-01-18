from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
def _format_nested_properties(self, properties: List[APIRequestBodyProperty], indent: int=2) -> str:
    """Format nested properties."""
    formatted_props = []
    for prop in properties:
        prop_name = prop.name
        prop_type = self.ts_type_from_python(prop.type)
        prop_required = '' if prop.required else '?'
        prop_desc = f'/* {prop.description} */' if prop.description else ''
        if prop.properties:
            nested_props = self._format_nested_properties(prop.properties, indent + 2)
            prop_type = f'{{\n{nested_props}\n{' ' * indent}}}'
        formatted_props.append(f'{prop_desc}\n{' ' * indent}{prop_name}{prop_required}: {prop_type},')
    return '\n'.join(formatted_props)