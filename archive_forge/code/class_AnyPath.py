import os
from abc import ABC
from pathlib import Path
from typing import Any, Union
from .cloudpath import InvalidPrefixError, CloudPath
from .exceptions import AnyPathTypeError
class AnyPath(ABC):
    """Polymorphic virtual superclass for CloudPath and pathlib.Path. Constructing an instance will
    automatically dispatch to CloudPath or Path based on the input. It also supports both
    isinstance and issubclass checks.

    This class also integrates with Pydantic. When used as a type declaration for a Pydantic
    BaseModel, the Pydantic validation process will appropriately run inputs through this class'
    constructor and dispatch to CloudPath or Path.
    """

    def __new__(cls, *args, **kwargs) -> Union[CloudPath, Path]:
        try:
            return CloudPath(*args, **kwargs)
        except InvalidPrefixError as cloudpath_exception:
            try:
                return Path(*args, **kwargs)
            except TypeError as path_exception:
                raise AnyPathTypeError(f'Invalid input for both CloudPath and Path. CloudPath exception: {repr(cloudpath_exception)} Path exception: {repr(path_exception)}')

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler):
        """Pydantic special method. See
        https://docs.pydantic.dev/2.0/usage/types/custom/"""
        try:
            from pydantic_core import core_schema
            return core_schema.no_info_after_validator_function(cls.validate, core_schema.any_schema())
        except ImportError:
            return None

    @classmethod
    def validate(cls, v: str) -> Union[CloudPath, Path]:
        """Pydantic special method. See
        https://docs.pydantic.dev/2.0/usage/types/custom/"""
        try:
            return cls.__new__(cls, v)
        except AnyPathTypeError as e:
            raise ValueError(e)

    @classmethod
    def __get_validators__(cls):
        """Pydantic special method. See
        https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types"""
        yield cls._validate

    @classmethod
    def _validate(cls, value) -> Union[CloudPath, Path]:
        """Used as a Pydantic validator. See
        https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types"""
        return cls.__new__(cls, value)