from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _infer_discriminator_values_for_inner_schema(self, schema: core_schema.CoreSchema, source: str) -> list[str | int]:
    """When inferring discriminator values for a field, we typically extract the expected values from a literal
        schema. This function does that, but also handles nested unions and defaults.
        """
    if schema['type'] == 'literal':
        return schema['expected']
    elif schema['type'] == 'union':
        values: list[Any] = []
        for choice in schema['choices']:
            choice_schema = choice[0] if isinstance(choice, tuple) else choice
            choice_values = self._infer_discriminator_values_for_inner_schema(choice_schema, source)
            values.extend(choice_values)
        return values
    elif schema['type'] == 'default':
        return self._infer_discriminator_values_for_inner_schema(schema['schema'], source)
    elif schema['type'] == 'function-after':
        return self._infer_discriminator_values_for_inner_schema(schema['schema'], source)
    elif schema['type'] in {'function-before', 'function-wrap', 'function-plain'}:
        validator_type = repr(schema['type'].split('-')[1])
        raise PydanticUserError(f'Cannot use a mode={validator_type} validator in the discriminator field {self.discriminator!r} of {source}', code='discriminator-validator')
    else:
        raise PydanticUserError(f'{source} needs field {self.discriminator!r} to be of type `Literal`', code='discriminator-needs-literal')