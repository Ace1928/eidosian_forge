from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
@dataclass(**slots_true)
class FieldSerializerDecoratorInfo:
    """A container for data from `@field_serializer` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@field_serializer'.
        fields: A tuple of field names the serializer should be called on.
        mode: The proposed serializer mode.
        return_type: The type of the serializer's return value.
        when_used: The serialization condition. Accepts a string with values `'always'`, `'unless-none'`, `'json'`,
            and `'json-unless-none'`.
        check_fields: Whether to check that the fields actually exist on the model.
    """
    decorator_repr: ClassVar[str] = '@field_serializer'
    fields: tuple[str, ...]
    mode: Literal['plain', 'wrap']
    return_type: Any
    when_used: core_schema.WhenUsed
    check_fields: bool | None