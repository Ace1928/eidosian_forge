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
def _extract_field_schema_pv2(model: type[BaseModel], field_name: str) -> ModelField | None:
    schema = model.__pydantic_core_schema__
    if schema['type'] != 'model':
        return None
    fields_schema = schema['schema']
    if fields_schema['type'] != 'model-fields':
        return None
    fields_schema = cast('ModelFieldsSchema', fields_schema)
    field = fields_schema['fields'].get(field_name)
    if not field:
        return None
    return cast('ModelField', field)