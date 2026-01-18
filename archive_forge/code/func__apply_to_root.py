from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _apply_to_root(self, schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    """This method handles the outer-most stage of recursion over the input schema:
        unwrapping nullable or definitions schemas, and calling the `_handle_choice`
        method iteratively on the choices extracted (recursively) from the possibly-wrapped union.
        """
    if schema['type'] == 'nullable':
        self._is_nullable = True
        wrapped = self._apply_to_root(schema['schema'])
        nullable_wrapper = schema.copy()
        nullable_wrapper['schema'] = wrapped
        return nullable_wrapper
    if schema['type'] == 'definitions':
        wrapped = self._apply_to_root(schema['schema'])
        definitions_wrapper = schema.copy()
        definitions_wrapper['schema'] = wrapped
        return definitions_wrapper
    if schema['type'] != 'union':
        schema = core_schema.union_schema([schema])
    choices_schemas = [v[0] if isinstance(v, tuple) else v for v in schema['choices'][::-1]]
    self._choices_to_handle.extend(choices_schemas)
    while self._choices_to_handle:
        choice = self._choices_to_handle.pop()
        self._handle_choice(choice)
    if self._discriminator_alias is not None and self._discriminator_alias != self.discriminator:
        discriminator: str | list[list[str | int]] = [[self.discriminator], [self._discriminator_alias]]
    else:
        discriminator = self.discriminator
    return core_schema.tagged_union_schema(choices=self._tagged_union_choices, discriminator=discriminator, custom_error_type=schema.get('custom_error_type'), custom_error_message=schema.get('custom_error_message'), custom_error_context=schema.get('custom_error_context'), strict=False, from_attributes=True, ref=schema.get('ref'), metadata=schema.get('metadata'), serialization=schema.get('serialization'))