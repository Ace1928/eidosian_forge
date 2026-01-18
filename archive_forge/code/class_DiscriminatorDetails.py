from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
class DiscriminatorDetails:
    field_name: str
    "The name of the discriminator field in the variant class, e.g.\n\n    ```py\n    class Foo(BaseModel):\n        type: Literal['foo']\n    ```\n\n    Will result in field_name='type'\n    "
    field_alias_from: str | None
    "The name of the discriminator field in the API response, e.g.\n\n    ```py\n    class Foo(BaseModel):\n        type: Literal['foo'] = Field(alias='type_from_api')\n    ```\n\n    Will result in field_alias_from='type_from_api'\n    "
    mapping: dict[str, type]
    "Mapping of discriminator value to variant type, e.g.\n\n    {'foo': FooVariant, 'bar': BarVariant}\n    "

    def __init__(self, *, mapping: dict[str, type], discriminator_field: str, discriminator_alias: str | None) -> None:
        self.mapping = mapping
        self.field_name = discriminator_field
        self.field_alias_from = discriminator_alias