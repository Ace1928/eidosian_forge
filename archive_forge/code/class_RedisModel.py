from __future__ import annotations
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing_extensions import TYPE_CHECKING, Literal
from langchain_community.vectorstores.redis.constants import REDIS_VECTOR_DTYPE_MAP
class RedisModel(BaseModel):
    """Schema for Redis index."""
    text: List[TextFieldSchema] = [TextFieldSchema(name='content')]
    tag: Optional[List[TagFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    extra: Optional[List[RedisField]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None
    content_key: str = 'content'
    content_vector_key: str = 'content_vector'

    def add_content_field(self) -> None:
        if self.text is None:
            self.text = []
        for field in self.text:
            if field.name == self.content_key:
                return
        self.text.append(TextFieldSchema(name=self.content_key))

    def add_vector_field(self, vector_field: Dict[str, Any]) -> None:
        if self.vector is None:
            self.vector = []
        if vector_field['algorithm'] == 'FLAT':
            self.vector.append(FlatVectorField(**vector_field))
        elif vector_field['algorithm'] == 'HNSW':
            self.vector.append(HNSWVectorField(**vector_field))
        else:
            raise ValueError(f'algorithm must be either FLAT or HNSW. Got {vector_field['algorithm']}')

    def as_dict(self) -> Dict[str, List[Any]]:
        schemas: Dict[str, List[Any]] = {'text': [], 'tag': [], 'numeric': []}
        for attr, attr_value in self.__dict__.items():
            if isinstance(attr_value, list) and len(attr_value) > 0:
                field_values: List[Dict[str, Any]] = []
                for val in attr_value:
                    value: Dict[str, Any] = {}
                    for field, field_value in val.__dict__.items():
                        if isinstance(field_value, Enum):
                            value[field] = field_value.value
                        elif field_value is not None:
                            value[field] = field_value
                    field_values.append(value)
                schemas[attr] = field_values
        schema: Dict[str, List[Any]] = {}
        for k, v in schemas.items():
            if len(v) > 0:
                schema[k] = v
        return schema

    @property
    def content_vector(self) -> Union[FlatVectorField, HNSWVectorField]:
        if not self.vector:
            raise ValueError('No vector fields found')
        for field in self.vector:
            if field.name == self.content_vector_key:
                return field
        raise ValueError('No content_vector field found')

    @property
    def vector_dtype(self) -> np.dtype:
        return REDIS_VECTOR_DTYPE_MAP[self.content_vector.datatype]

    @property
    def is_empty(self) -> bool:
        return all((field is None for field in [self.tag, self.text, self.numeric, self.vector]))

    def get_fields(self) -> List['RedisField']:
        redis_fields: List['RedisField'] = []
        if self.is_empty:
            return redis_fields
        for field_name in self.__fields__.keys():
            if field_name not in ['content_key', 'content_vector_key', 'extra']:
                field_group = getattr(self, field_name)
                if field_group is not None:
                    for field in field_group:
                        redis_fields.append(field.as_field())
        return redis_fields

    @property
    def metadata_keys(self) -> List[str]:
        keys: List[str] = []
        if self.is_empty:
            return keys
        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    if not isinstance(field, str) and field.name not in [self.content_key, self.content_vector_key]:
                        keys.append(field.name)
        return keys