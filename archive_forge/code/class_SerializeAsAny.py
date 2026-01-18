from __future__ import annotations
import dataclasses
from functools import partialmethod
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from pydantic_core import PydanticUndefined, core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import PydanticUndefinedAnnotation
from ._internal import _decorators, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class SerializeAsAny:

    def __class_getitem__(cls, item: Any) -> Any:
        return Annotated[item, SerializeAsAny()]

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema = handler(source_type)
        schema_to_update = schema
        while schema_to_update['type'] == 'definitions':
            schema_to_update = schema_to_update.copy()
            schema_to_update = schema_to_update['schema']
        schema_to_update['serialization'] = core_schema.wrap_serializer_function_ser_schema(lambda x, h: h(x), schema=core_schema.any_schema())
        return schema
    __hash__ = object.__hash__