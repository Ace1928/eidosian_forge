from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def _infer_discriminator_values_for_choice(self, choice: core_schema.CoreSchema, source_name: str | None) -> list[str | int]:
    """This function recurses over `choice`, extracting all discriminator values that should map to this choice.

        `model_name` is accepted for the purpose of producing useful error messages.
        """
    if choice['type'] == 'definitions':
        return self._infer_discriminator_values_for_choice(choice['schema'], source_name=source_name)
    elif choice['type'] == 'function-plain':
        raise TypeError(f'{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`')
    elif _core_utils.is_function_with_inner_schema(choice):
        return self._infer_discriminator_values_for_choice(choice['schema'], source_name=source_name)
    elif choice['type'] == 'lax-or-strict':
        return sorted(set(self._infer_discriminator_values_for_choice(choice['lax_schema'], source_name=None) + self._infer_discriminator_values_for_choice(choice['strict_schema'], source_name=None)))
    elif choice['type'] == 'tagged-union':
        values: list[str | int] = []
        subchoices = [x for x in choice['choices'].values() if not isinstance(x, (str, int))]
        for subchoice in subchoices:
            subchoice_values = self._infer_discriminator_values_for_choice(subchoice, source_name=None)
            values.extend(subchoice_values)
        return values
    elif choice['type'] == 'union':
        values = []
        for subchoice in choice['choices']:
            subchoice_schema = subchoice[0] if isinstance(subchoice, tuple) else subchoice
            subchoice_values = self._infer_discriminator_values_for_choice(subchoice_schema, source_name=None)
            values.extend(subchoice_values)
        return values
    elif choice['type'] == 'nullable':
        self._should_be_nullable = True
        return self._infer_discriminator_values_for_choice(choice['schema'], source_name=None)
    elif choice['type'] == 'model':
        return self._infer_discriminator_values_for_choice(choice['schema'], source_name=choice['cls'].__name__)
    elif choice['type'] == 'dataclass':
        return self._infer_discriminator_values_for_choice(choice['schema'], source_name=choice['cls'].__name__)
    elif choice['type'] == 'model-fields':
        return self._infer_discriminator_values_for_model_choice(choice, source_name=source_name)
    elif choice['type'] == 'dataclass-args':
        return self._infer_discriminator_values_for_dataclass_choice(choice, source_name=source_name)
    elif choice['type'] == 'typed-dict':
        return self._infer_discriminator_values_for_typed_dict_choice(choice, source_name=source_name)
    elif choice['type'] == 'definition-ref':
        schema_ref = choice['schema_ref']
        if schema_ref not in self.definitions:
            raise MissingDefinitionForUnionRef(schema_ref)
        return self._infer_discriminator_values_for_choice(self.definitions[schema_ref], source_name=source_name)
    else:
        raise TypeError(f'{choice['type']!r} is not a valid discriminated union variant; should be a `BaseModel` or `dataclass`')