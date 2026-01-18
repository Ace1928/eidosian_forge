from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _infer_discriminator_values_for_typed_dict_choice(self, choice: core_schema.TypedDictSchema, source_name: str | None=None) -> list[str | int]:
    """This method just extracts the _infer_discriminator_values_for_choice logic specific to TypedDictSchema
        for the sake of readability.
        """
    source = 'TypedDict' if source_name is None else f'TypedDict {source_name!r}'
    field = choice['fields'].get(self.discriminator)
    if field is None:
        raise PydanticUserError(f'{source} needs a discriminator field for key {self.discriminator!r}', code='discriminator-no-field')
    return self._infer_discriminator_values_for_field(field, source)