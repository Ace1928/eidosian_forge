from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fastapi._compat import (
from fastapi.logger import logger
from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated
class Schema(BaseModel):
    schema_: Optional[str] = Field(default=None, alias='$schema')
    vocabulary: Optional[str] = Field(default=None, alias='$vocabulary')
    id: Optional[str] = Field(default=None, alias='$id')
    anchor: Optional[str] = Field(default=None, alias='$anchor')
    dynamicAnchor: Optional[str] = Field(default=None, alias='$dynamicAnchor')
    ref: Optional[str] = Field(default=None, alias='$ref')
    dynamicRef: Optional[str] = Field(default=None, alias='$dynamicRef')
    defs: Optional[Dict[str, 'SchemaOrBool']] = Field(default=None, alias='$defs')
    comment: Optional[str] = Field(default=None, alias='$comment')
    allOf: Optional[List['SchemaOrBool']] = None
    anyOf: Optional[List['SchemaOrBool']] = None
    oneOf: Optional[List['SchemaOrBool']] = None
    not_: Optional['SchemaOrBool'] = Field(default=None, alias='not')
    if_: Optional['SchemaOrBool'] = Field(default=None, alias='if')
    then: Optional['SchemaOrBool'] = None
    else_: Optional['SchemaOrBool'] = Field(default=None, alias='else')
    dependentSchemas: Optional[Dict[str, 'SchemaOrBool']] = None
    prefixItems: Optional[List['SchemaOrBool']] = None
    items: Optional[Union['SchemaOrBool', List['SchemaOrBool']]] = None
    contains: Optional['SchemaOrBool'] = None
    properties: Optional[Dict[str, 'SchemaOrBool']] = None
    patternProperties: Optional[Dict[str, 'SchemaOrBool']] = None
    additionalProperties: Optional['SchemaOrBool'] = None
    propertyNames: Optional['SchemaOrBool'] = None
    unevaluatedItems: Optional['SchemaOrBool'] = None
    unevaluatedProperties: Optional['SchemaOrBool'] = None
    type: Optional[str] = None
    enum: Optional[List[Any]] = None
    const: Optional[Any] = None
    multipleOf: Optional[float] = Field(default=None, gt=0)
    maximum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    minimum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    maxLength: Optional[int] = Field(default=None, ge=0)
    minLength: Optional[int] = Field(default=None, ge=0)
    pattern: Optional[str] = None
    maxItems: Optional[int] = Field(default=None, ge=0)
    minItems: Optional[int] = Field(default=None, ge=0)
    uniqueItems: Optional[bool] = None
    maxContains: Optional[int] = Field(default=None, ge=0)
    minContains: Optional[int] = Field(default=None, ge=0)
    maxProperties: Optional[int] = Field(default=None, ge=0)
    minProperties: Optional[int] = Field(default=None, ge=0)
    required: Optional[List[str]] = None
    dependentRequired: Optional[Dict[str, Set[str]]] = None
    format: Optional[str] = None
    contentEncoding: Optional[str] = None
    contentMediaType: Optional[str] = None
    contentSchema: Optional['SchemaOrBool'] = None
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    deprecated: Optional[bool] = None
    readOnly: Optional[bool] = None
    writeOnly: Optional[bool] = None
    examples: Optional[List[Any]] = None
    discriminator: Optional[Discriminator] = None
    xml: Optional[XML] = None
    externalDocs: Optional[ExternalDocumentation] = None
    example: Annotated[Optional[Any], typing_deprecated('Deprecated in OpenAPI 3.1.0 that now uses JSON Schema 2020-12, although still supported. Use examples instead.')] = None
    if PYDANTIC_V2:
        model_config = {'extra': 'allow'}
    else:

        class Config:
            extra = 'allow'