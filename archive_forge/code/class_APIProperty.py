from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
class APIProperty(APIPropertyBase):
    """A model for a property in the query, path, header, or cookie params."""
    location: APIPropertyLocation = Field(alias='location')
    "The path/how it's being passed to the endpoint."

    @staticmethod
    def _cast_schema_list_type(schema: Schema) -> Optional[Union[str, Tuple[str, ...]]]:
        type_ = schema.type
        if not isinstance(type_, list):
            return type_
        else:
            return tuple(type_)

    @staticmethod
    def _get_schema_type_for_enum(parameter: Parameter, schema: Schema) -> Enum:
        """Get the schema type when the parameter is an enum."""
        param_name = f'{parameter.name}Enum'
        return Enum(param_name, {str(v): v for v in schema.enum})

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

    @staticmethod
    def _validate_location(location: APIPropertyLocation, name: str) -> None:
        if location not in SUPPORTED_LOCATIONS:
            raise NotImplementedError(INVALID_LOCATION_TEMPL.format(location=location, name=name))

    @staticmethod
    def _validate_content(content: Optional[Dict[str, MediaType]]) -> None:
        if content:
            raise ValueError("API Properties with media content not supported. Media content only supported within APIRequestBodyProperty's")

    @staticmethod
    def _get_schema(parameter: Parameter, spec: OpenAPISpec) -> Optional[Schema]:
        from openapi_pydantic import Reference, Schema
        schema = parameter.param_schema
        if isinstance(schema, Reference):
            schema = spec.get_referenced_schema(schema)
        elif schema is None:
            return None
        elif not isinstance(schema, Schema):
            raise ValueError(f'Error dereferencing schema: {schema}')
        return schema

    @staticmethod
    def is_supported_location(location: str) -> bool:
        """Return whether the provided location is supported."""
        try:
            return APIPropertyLocation.from_str(location) in SUPPORTED_LOCATIONS
        except ValueError:
            return False

    @classmethod
    def from_parameter(cls, parameter: Parameter, spec: OpenAPISpec) -> 'APIProperty':
        """Instantiate from an OpenAPI Parameter."""
        location = APIPropertyLocation.from_str(parameter.param_in)
        cls._validate_location(location, parameter.name)
        cls._validate_content(parameter.content)
        schema = cls._get_schema(parameter, spec)
        schema_type = cls._get_schema_type(parameter, schema)
        default_val = schema.default if schema is not None else None
        return cls(name=parameter.name, location=location, default=default_val, description=parameter.description, required=parameter.required, type=schema_type)