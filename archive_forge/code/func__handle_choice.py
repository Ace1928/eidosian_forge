from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _handle_choice(self, choice: core_schema.CoreSchema) -> None:
    """This method handles the "middle" stage of recursion over the input schema.
        Specifically, it is responsible for handling each choice of the outermost union
        (and any "coalesced" choices obtained from inner unions).

        Here, "handling" entails:
        * Coalescing nested unions and compatible tagged-unions
        * Tracking the presence of 'none' and 'nullable' schemas occurring as choices
        * Validating that each allowed discriminator value maps to a unique choice
        * Updating the _tagged_union_choices mapping that will ultimately be used to build the TaggedUnionSchema.
        """
    if choice['type'] == 'definition-ref':
        if choice['schema_ref'] not in self.definitions:
            raise MissingDefinitionForUnionRef(choice['schema_ref'])
    if choice['type'] == 'none':
        self._should_be_nullable = True
    elif choice['type'] == 'definitions':
        self._handle_choice(choice['schema'])
    elif choice['type'] == 'nullable':
        self._should_be_nullable = True
        self._handle_choice(choice['schema'])
    elif choice['type'] == 'union':
        choices_schemas = [v[0] if isinstance(v, tuple) else v for v in choice['choices'][::-1]]
        self._choices_to_handle.extend(choices_schemas)
    elif choice['type'] not in {'model', 'typed-dict', 'tagged-union', 'lax-or-strict', 'dataclass', 'dataclass-args', 'definition-ref'} and (not _core_utils.is_function_with_inner_schema(choice)):
        raise TypeError(f'{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`')
    else:
        if choice['type'] == 'tagged-union' and self._is_discriminator_shared(choice):
            subchoices = [x for x in choice['choices'].values() if not isinstance(x, (str, int))]
            self._choices_to_handle.extend(subchoices[::-1])
            return
        inferred_discriminator_values = self._infer_discriminator_values_for_choice(choice, source_name=None)
        self._set_unique_choice_for_values(choice, inferred_discriminator_values)