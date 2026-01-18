from __future__ import annotations as _annotations
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import MockValSer, set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute, SafeGetItemProxy
from ._validate_call import ValidateCallWrapper
@staticmethod
def _collect_bases_data(bases: tuple[type[Any], ...]) -> tuple[set[str], set[str], dict[str, ModelPrivateAttr]]:
    from ..main import BaseModel
    field_names: set[str] = set()
    class_vars: set[str] = set()
    private_attributes: dict[str, ModelPrivateAttr] = {}
    for base in bases:
        if issubclass(base, BaseModel) and base is not BaseModel:
            field_names.update(getattr(base, 'model_fields', {}).keys())
            class_vars.update(base.__class_vars__)
            private_attributes.update(base.__private_attributes__)
    return (field_names, class_vars, private_attributes)