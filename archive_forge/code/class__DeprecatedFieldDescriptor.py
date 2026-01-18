from __future__ import annotations as _annotations
import builtins
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic, NoReturn
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases, unwrap_wrapped_function
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import MockValSer, set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute, SafeGetItemProxy
from ._validate_call import ValidateCallWrapper
class _DeprecatedFieldDescriptor:
    """Data descriptor used to emit a runtime deprecation warning before accessing a deprecated field.

    Attributes:
        msg: The deprecation message to be emitted.
        wrapped_property: The property instance if the deprecated field is a computed field, or `None`.
        field_name: The name of the field being deprecated.
    """
    field_name: str

    def __init__(self, msg: str, wrapped_property: property | None=None) -> None:
        self.msg = msg
        self.wrapped_property = wrapped_property

    def __set_name__(self, cls: type[BaseModel], name: str) -> None:
        self.field_name = name

    def __get__(self, obj: BaseModel | None, obj_type: type[BaseModel] | None=None) -> Any:
        if obj is None:
            raise AttributeError(self.field_name)
        warnings.warn(self.msg, builtins.DeprecationWarning, stacklevel=2)
        if self.wrapped_property is not None:
            return self.wrapped_property.__get__(obj, obj_type)
        return obj.__dict__[self.field_name]

    def __set__(self, obj: Any, value: Any) -> NoReturn:
        raise AttributeError(self.field_name)