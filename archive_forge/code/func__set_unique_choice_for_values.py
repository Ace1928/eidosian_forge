from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _set_unique_choice_for_values(self, choice: core_schema.CoreSchema, values: Sequence[str | int]) -> None:
    """This method updates `self.tagged_union_choices` so that all provided (discriminator) `values` map to the
        provided `choice`, validating that none of these values already map to another (different) choice.
        """
    for discriminator_value in values:
        if discriminator_value in self._tagged_union_choices:
            existing_choice = self._tagged_union_choices[discriminator_value]
            if existing_choice != choice:
                raise TypeError(f'Value {discriminator_value!r} for discriminator {self.discriminator!r} mapped to multiple choices')
        else:
            self._tagged_union_choices[discriminator_value] = choice