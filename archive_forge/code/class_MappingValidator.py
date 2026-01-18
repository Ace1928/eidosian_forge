from __future__ import annotations as _annotations
import collections
import collections.abc
import dataclasses
import decimal
import inspect
import os
import typing
from enum import Enum
from functools import partial
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any, Callable, Iterable, TypeVar
import typing_extensions
from pydantic_core import (
from typing_extensions import get_args, get_origin
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.types import Strict
from ..config import ConfigDict
from ..json_schema import JsonSchemaValue, update_json_schema
from . import _known_annotated_metadata, _typing_extra, _validators
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._schema_generation_shared import GetCoreSchemaHandler, GetJsonSchemaHandler
@dataclasses.dataclass(**slots_true)
class MappingValidator:
    mapped_origin: type[Any]
    keys_source_type: type[Any]
    values_source_type: type[Any]
    min_length: int | None = None
    max_length: int | None = None
    strict: bool = False

    def serialize_mapping_via_dict(self, v: Any, handler: core_schema.SerializerFunctionWrapHandler) -> Any:
        return handler(v)

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        if self.keys_source_type is Any:
            keys_schema = None
        else:
            keys_schema = handler.generate_schema(self.keys_source_type)
        if self.values_source_type is Any:
            values_schema = None
        else:
            values_schema = handler.generate_schema(self.values_source_type)
        metadata = {'min_length': self.min_length, 'max_length': self.max_length, 'strict': self.strict}
        if self.mapped_origin is dict:
            schema = core_schema.dict_schema(keys_schema, values_schema, **metadata)
        else:
            constrained_schema = core_schema.dict_schema(keys_schema, values_schema, **metadata)
            check_instance = core_schema.json_or_python_schema(json_schema=core_schema.dict_schema(), python_schema=core_schema.is_instance_schema(self.mapped_origin))
            if self.mapped_origin is collections.defaultdict:
                default_default_factory = get_defaultdict_default_default_factory(self.values_source_type)
                coerce_instance_wrap = partial(core_schema.no_info_wrap_validator_function, partial(defaultdict_validator, default_default_factory=default_default_factory))
            else:
                coerce_instance_wrap = partial(core_schema.no_info_after_validator_function, self.mapped_origin)
            serialization = core_schema.wrap_serializer_function_ser_schema(self.serialize_mapping_via_dict, schema=core_schema.dict_schema(keys_schema or core_schema.any_schema(), values_schema or core_schema.any_schema()), info_arg=False)
            strict = core_schema.chain_schema([check_instance, coerce_instance_wrap(constrained_schema)])
            if metadata.get('strict', False):
                schema = strict
            else:
                lax = coerce_instance_wrap(constrained_schema)
                schema = core_schema.lax_or_strict_schema(lax_schema=lax, strict_schema=strict)
                schema['serialization'] = serialization
        return schema