import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from .class_validators import gather_all_validators
from .config import BaseConfig, ConfigDict, Extra, get_config
from .error_wrappers import ValidationError
from .errors import DataclassTypeError
from .fields import Field, FieldInfo, Required, Undefined
from .main import create_model, validate_model
from .utils import ClassAttribute
class Dataclass:
    __dataclass_fields__: ClassVar[Dict[str, Any]]
    __dataclass_params__: ClassVar[Any]
    __post_init__: ClassVar[Callable[..., None]]
    __pydantic_run_validation__: ClassVar[bool]
    __post_init_post_parse__: ClassVar[Callable[..., None]]
    __pydantic_initialised__: ClassVar[bool]
    __pydantic_model__: ClassVar[Type[BaseModel]]
    __pydantic_validate_values__: ClassVar[Callable[['Dataclass'], None]]
    __pydantic_has_field_info_default__: ClassVar[bool]

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    @classmethod
    def __get_validators__(cls: Type['Dataclass']) -> 'CallableGenerator':
        pass

    @classmethod
    def __validate__(cls: Type['DataclassT'], v: Any) -> 'DataclassT':
        pass