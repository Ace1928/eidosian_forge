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
class SequenceValidator:
    mapped_origin: type[Any]
    item_source_type: type[Any]
    min_length: int | None = None
    max_length: int | None = None
    strict: bool = False

    def serialize_sequence_via_list(self, v: Any, handler: core_schema.SerializerFunctionWrapHandler, info: core_schema.SerializationInfo) -> Any:
        items: list[Any] = []
        for index, item in enumerate(v):
            try:
                v = handler(item, index)
            except PydanticOmit:
                pass
            else:
                items.append(v)
        if info.mode_is_json():
            return items
        else:
            return self.mapped_origin(items)

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        if self.item_source_type is Any:
            items_schema = None
        else:
            items_schema = handler.generate_schema(self.item_source_type)
        metadata = {'min_length': self.min_length, 'max_length': self.max_length, 'strict': self.strict}
        if self.mapped_origin in (list, set, frozenset):
            if self.mapped_origin is list:
                constrained_schema = core_schema.list_schema(items_schema, **metadata)
            elif self.mapped_origin is set:
                constrained_schema = core_schema.set_schema(items_schema, **metadata)
            else:
                assert self.mapped_origin is frozenset
                constrained_schema = core_schema.frozenset_schema(items_schema, **metadata)
            schema = constrained_schema
        else:
            assert self.mapped_origin in (collections.deque, collections.Counter)
            if self.mapped_origin is collections.deque:
                coerce_instance_wrap = partial(core_schema.no_info_wrap_validator_function, partial(dequeue_validator, maxlen=metadata.get('max_length', None)))
            else:
                coerce_instance_wrap = partial(core_schema.no_info_after_validator_function, self.mapped_origin)
            constrained_schema = core_schema.list_schema(items_schema, **metadata)
            check_instance = core_schema.json_or_python_schema(json_schema=core_schema.list_schema(), python_schema=core_schema.is_instance_schema(self.mapped_origin))
            serialization = core_schema.wrap_serializer_function_ser_schema(self.serialize_sequence_via_list, schema=items_schema or core_schema.any_schema(), info_arg=True)
            strict = core_schema.chain_schema([check_instance, coerce_instance_wrap(constrained_schema)])
            if metadata.get('strict', False):
                schema = strict
            else:
                lax = coerce_instance_wrap(constrained_schema)
                schema = core_schema.lax_or_strict_schema(lax_schema=lax, strict_schema=strict)
            schema['serialization'] = serialization
        return schema