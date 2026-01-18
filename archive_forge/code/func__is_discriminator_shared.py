from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _is_discriminator_shared(self, choice: core_schema.TaggedUnionSchema) -> bool:
    """This method returns a boolean indicating whether the discriminator for the `choice`
        is the same as that being used for the outermost tagged union. This is used to
        determine whether this TaggedUnionSchema choice should be "coalesced" into the top level,
        or whether it should be treated as a separate (nested) choice.
        """
    inner_discriminator = choice['discriminator']
    return inner_discriminator == self.discriminator or (isinstance(inner_discriminator, list) and (self.discriminator in inner_discriminator or [self.discriminator] in inner_discriminator))