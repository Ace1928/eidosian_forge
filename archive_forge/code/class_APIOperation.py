from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
class APIOperation(BaseModel):
    """A model for a single API operation."""
    operation_id: str = Field(alias='operation_id')
    'The unique identifier of the operation.'
    description: Optional[str] = Field(alias='description')
    'The description of the operation.'
    base_url: str = Field(alias='base_url')
    'The base URL of the operation.'
    path: str = Field(alias='path')
    'The path of the operation.'
    method: HTTPVerb = Field(alias='method')
    'The HTTP method of the operation.'
    properties: Sequence[APIProperty] = Field(alias='properties')
    request_body: Optional[APIRequestBody] = Field(alias='request_body')
    'The request body of the operation.'

    @staticmethod
    def _get_properties_from_parameters(parameters: List[Parameter], spec: OpenAPISpec) -> List[APIProperty]:
        """Get the properties of the operation."""
        properties = []
        for param in parameters:
            if APIProperty.is_supported_location(param.param_in):
                properties.append(APIProperty.from_parameter(param, spec))
            elif param.required:
                raise ValueError(INVALID_LOCATION_TEMPL.format(location=param.param_in, name=param.name))
            else:
                logger.warning(INVALID_LOCATION_TEMPL.format(location=param.param_in, name=param.name) + ' Ignoring optional parameter')
                pass
        return properties

    @classmethod
    def from_openapi_url(cls, spec_url: str, path: str, method: str) -> 'APIOperation':
        """Create an APIOperation from an OpenAPI URL."""
        spec = OpenAPISpec.from_url(spec_url)
        return cls.from_openapi_spec(spec, path, method)

    @classmethod
    def from_openapi_spec(cls, spec: OpenAPISpec, path: str, method: str) -> 'APIOperation':
        """Create an APIOperation from an OpenAPI spec."""
        operation = spec.get_operation(path, method)
        parameters = spec.get_parameters_for_operation(operation)
        properties = cls._get_properties_from_parameters(parameters, spec)
        operation_id = OpenAPISpec.get_cleaned_operation_id(operation, path, method)
        request_body = spec.get_request_body_for_operation(operation)
        api_request_body = APIRequestBody.from_request_body(request_body, spec) if request_body is not None else None
        description = operation.description or operation.summary
        if not description and spec.paths is not None:
            description = spec.paths[path].description or spec.paths[path].summary
        return cls(operation_id=operation_id, description=description or '', base_url=spec.base_url, path=path, method=method, properties=properties, request_body=api_request_body)

    @staticmethod
    def ts_type_from_python(type_: SCHEMA_TYPE) -> str:
        if type_ is None:
            return 'any'
        elif isinstance(type_, str):
            return {'str': 'string', 'integer': 'number', 'float': 'number', 'date-time': 'string'}.get(type_, type_)
        elif isinstance(type_, tuple):
            return f'Array<{APIOperation.ts_type_from_python(type_[0])}>'
        elif isinstance(type_, type) and issubclass(type_, Enum):
            return ' | '.join([f"'{e.value}'" for e in type_])
        else:
            return str(type_)

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

    def to_typescript(self) -> str:
        """Get typescript string representation of the operation."""
        operation_name = self.operation_id
        params = []
        if self.request_body:
            formatted_request_body_props = self._format_nested_properties(self.request_body.properties)
            params.append(formatted_request_body_props)
        for prop in self.properties:
            prop_name = prop.name
            prop_type = self.ts_type_from_python(prop.type)
            prop_required = '' if prop.required else '?'
            prop_desc = f'/* {prop.description} */' if prop.description else ''
            params.append(f'{prop_desc}\n\t\t{prop_name}{prop_required}: {prop_type},')
        formatted_params = '\n'.join(params).strip()
        description_str = f'/* {self.description} */' if self.description else ''
        typescript_definition = f'\n{description_str}\ntype {operation_name} = (_: {{\n{formatted_params}\n}}) => any;\n'
        return typescript_definition.strip()

    @property
    def query_params(self) -> List[str]:
        return [property.name for property in self.properties if property.location == APIPropertyLocation.QUERY]

    @property
    def path_params(self) -> List[str]:
        return [property.name for property in self.properties if property.location == APIPropertyLocation.PATH]

    @property
    def body_params(self) -> List[str]:
        if self.request_body is None:
            return []
        return [prop.name for prop in self.request_body.properties]