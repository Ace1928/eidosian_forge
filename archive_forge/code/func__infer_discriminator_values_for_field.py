from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _infer_discriminator_values_for_field(self, field: CoreSchemaField, source: str) -> list[str | int]:
    if field['type'] == 'computed-field':
        return []
    alias = field.get('validation_alias', self.discriminator)
    if not isinstance(alias, str):
        raise PydanticUserError(f'Alias {alias!r} is not supported in a discriminated union', code='discriminator-alias-type')
    if self._discriminator_alias is None:
        self._discriminator_alias = alias
    elif self._discriminator_alias != alias:
        raise PydanticUserError(f'Aliases for discriminator {self.discriminator!r} must be the same (got {alias}, {self._discriminator_alias})', code='discriminator-alias')
    return self._infer_discriminator_values_for_inner_schema(field['schema'], source)