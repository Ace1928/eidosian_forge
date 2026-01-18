from __future__ import annotations as _annotations
import dataclasses as _dataclasses
import re
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Any
from pydantic_core import MultiHostUrl, PydanticCustomError, Url, core_schema
from typing_extensions import Annotated, TypeAlias
from ._internal import _fields, _repr, _schema_generation_shared
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler
from .json_schema import JsonSchemaValue
from pydantic import BaseModel, HttpUrl, ValidationError
from pydantic import BaseModel, HttpUrl
from pydantic import (
class NameEmail(_repr.Representation):
    """
    Info:
        To use this type, you need to install the optional
        [`email-validator`](https://github.com/JoshData/python-email-validator) package:

        ```bash
        pip install email-validator
        ```

    Validate a name and email address combination, as specified by
    [RFC 5322](https://datatracker.ietf.org/doc/html/rfc5322#section-3.4).

    The `NameEmail` has two properties: `name` and `email`.
    In case the `name` is not provided, it's inferred from the email address.

    ```py
    from pydantic import BaseModel, NameEmail

    class User(BaseModel):
        email: NameEmail

    user = User(email='Fred Bloggs <fred.bloggs@example.com>')
    print(user.email)
    #> Fred Bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> Fred Bloggs

    user = User(email='fred.bloggs@example.com')
    print(user.email)
    #> fred.bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> fred.bloggs
    ```
    """
    __slots__ = ('name', 'email')

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        import_email_validator()
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.union_schema([core_schema.is_instance_schema(cls), core_schema.str_schema()], custom_error_type='name_email_type', custom_error_message='Input is not a valid NameEmail'), serialization=core_schema.to_string_ser_schema())

    @classmethod
    def _validate(cls, __input_value: NameEmail | str) -> NameEmail:
        if isinstance(__input_value, cls):
            return __input_value
        else:
            name, email = validate_email(__input_value)
            return cls(name, email)

    def __str__(self) -> str:
        return f'{self.name} <{self.email}>'