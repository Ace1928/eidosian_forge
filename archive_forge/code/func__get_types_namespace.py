from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import core_schema
from typing_extensions import Literal
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
def _get_types_namespace(self) -> dict[str, Any] | None:
    return self._generate_schema._types_namespace