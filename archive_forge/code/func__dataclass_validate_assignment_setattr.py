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
def _dataclass_validate_assignment_setattr(self: 'Dataclass', name: str, value: Any) -> None:
    if self.__pydantic_initialised__:
        d = dict(self.__dict__)
        d.pop(name, None)
        known_field = self.__pydantic_model__.__fields__.get(name, None)
        if known_field:
            value, error_ = known_field.validate(value, d, loc=name, cls=self.__class__)
            if error_:
                raise ValidationError([error_], self.__class__)
    object.__setattr__(self, name, value)