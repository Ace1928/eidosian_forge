import sys
import types
import typing
from typing import (
from weakref import WeakKeyDictionary, WeakValueDictionary
from typing_extensions import Annotated, Literal as ExtLiteral
from .class_validators import gather_all_validators
from .fields import DeferredType
from .main import BaseModel, create_model
from .types import JsonWrapper
from .typing import display_as_type, get_all_type_hints, get_args, get_origin, typing_base
from .utils import all_identical, lenient_issubclass
def _prepare_model_fields(created_model: Type[GenericModel], fields: Mapping[str, Any], instance_type_hints: Mapping[str, type], typevars_map: Mapping[Any, type]) -> None:
    """
    Replace DeferredType fields with concrete type hints and prepare them.
    """
    for key, field in created_model.__fields__.items():
        if key not in fields:
            assert field.type_.__class__ is not DeferredType
            continue
        assert field.type_.__class__ is DeferredType, field.type_.__class__
        field_type_hint = instance_type_hints[key]
        concrete_type = replace_types(field_type_hint, typevars_map)
        field.type_ = concrete_type
        field.outer_type_ = concrete_type
        field.prepare()
        created_model.__annotations__[key] = concrete_type