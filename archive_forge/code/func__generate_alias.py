from __future__ import annotations
import dataclasses
from typing import Callable, Literal
from ._internal import _internal_dataclass
def _generate_alias(self, alias_kind: Literal['alias', 'validation_alias', 'serialization_alias'], allowed_types: tuple[type[str] | type[AliasPath] | type[AliasChoices], ...], field_name: str) -> str | AliasPath | AliasChoices | None:
    """Generate an alias of the specified kind. Returns None if the alias generator is None.

        Raises:
            TypeError: If the alias generator produces an invalid type.
        """
    alias = None
    if (alias_generator := getattr(self, alias_kind)):
        alias = alias_generator(field_name)
        if alias and (not isinstance(alias, allowed_types)):
            raise TypeError(f'Invalid `{alias_kind}` type. `{alias_kind}` generator must produce one of `{allowed_types}`')
    return alias