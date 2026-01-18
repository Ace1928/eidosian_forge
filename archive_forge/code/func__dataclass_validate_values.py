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
def _dataclass_validate_values(self: 'Dataclass') -> None:
    if getattr(self, '__pydantic_initialised__'):
        return
    if getattr(self, '__pydantic_has_field_info_default__', False):
        input_data = {k: v for k, v in self.__dict__.items() if not (isinstance(v, FieldInfo) or _is_field_cached_property(self, k))}
    else:
        input_data = {k: v for k, v in self.__dict__.items() if not _is_field_cached_property(self, k)}
    d, _, validation_error = validate_model(self.__pydantic_model__, input_data, cls=self.__class__)
    if validation_error:
        raise validation_error
    self.__dict__.update(d)
    object.__setattr__(self, '__pydantic_initialised__', True)