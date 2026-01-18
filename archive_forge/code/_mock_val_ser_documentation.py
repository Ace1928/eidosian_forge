from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from pydantic_core import SchemaSerializer, SchemaValidator
from typing_extensions import Literal
from ..errors import PydanticErrorCodes, PydanticUserError
Set `__pydantic_validator__` and `__pydantic_serializer__` to `MockValSer`s on a dataclass.

    Args:
        cls: The model class to set the mocks on
        cls_name: Name of the model class, used in error messages
        undefined_name: Name of the undefined thing, used in error messages
    