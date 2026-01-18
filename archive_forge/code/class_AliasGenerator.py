from __future__ import annotations
import dataclasses
from typing import Callable, Literal
from ._internal import _internal_dataclass
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class AliasGenerator:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/alias#using-an-aliasgenerator

    A data class used by `alias_generator` as a convenience to create various aliases.

    Attributes:
        alias: A callable that takes a field name and returns an alias for it.
        validation_alias: A callable that takes a field name and returns a validation alias for it.
        serialization_alias: A callable that takes a field name and returns a serialization alias for it.
    """
    alias: Callable[[str], str] | None = None
    validation_alias: Callable[[str], str | AliasPath | AliasChoices] | None = None
    serialization_alias: Callable[[str], str] | None = None

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

    def generate_aliases(self, field_name: str) -> tuple[str | None, str | AliasPath | AliasChoices | None, str | None]:
        """Generate `alias`, `validation_alias`, and `serialization_alias` for a field.

        Returns:
            A tuple of three aliases - validation, alias, and serialization.
        """
        alias = self._generate_alias('alias', (str,), field_name)
        validation_alias = self._generate_alias('validation_alias', (str, AliasChoices, AliasPath), field_name)
        serialization_alias = self._generate_alias('serialization_alias', (str,), field_name)
        return (alias, validation_alias, serialization_alias)