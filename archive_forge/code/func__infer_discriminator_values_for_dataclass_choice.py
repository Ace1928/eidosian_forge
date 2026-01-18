from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _infer_discriminator_values_for_dataclass_choice(self, choice: core_schema.DataclassArgsSchema, source_name: str | None=None) -> list[str | int]:
    source = 'DataclassArgs' if source_name is None else f'Dataclass {source_name!r}'
    for field in choice['fields']:
        if field['name'] == self.discriminator:
            break
    else:
        raise PydanticUserError(f'{source} needs a discriminator field for key {self.discriminator!r}', code='discriminator-no-field')
    return self._infer_discriminator_values_for_field(field, source)