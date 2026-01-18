from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import PydanticUndefined
from ._config import ConfigWrapper
from ._utils import is_valid_identifier
def _field_name_for_signature(field_name: str, field_info: FieldInfo) -> str:
    """Extract the correct name to use for the field when generating a signature.

    Assuming the field has a valid alias, this will return the alias. Otherwise, it will return the field name.
    First priority is given to the validation_alias, then the alias, then the field name.

    Args:
        field_name: The name of the field
        field_info: The corresponding FieldInfo object.

    Returns:
        The correct name to use when generating a signature.
    """

    def _alias_if_valid(x: Any) -> str | None:
        """Return the alias if it is a valid alias and identifier, else None."""
        return x if isinstance(x, str) and is_valid_identifier(x) else None
    return _alias_if_valid(field_info.alias) or _alias_if_valid(field_info.validation_alias) or field_name