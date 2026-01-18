from __future__ import annotations
import dataclasses
from typing import Callable, Literal
from ._internal import _internal_dataclass
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class AliasPath:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/alias#aliaspath-and-aliaschoices

    A data class used by `validation_alias` as a convenience to create aliases.

    Attributes:
        path: A list of string or integer aliases.
    """
    path: list[int | str]

    def __init__(self, first_arg: str, *args: str | int) -> None:
        self.path = [first_arg] + list(args)

    def convert_to_aliases(self) -> list[str | int]:
        """Converts arguments to a list of string or integer aliases.

        Returns:
            The list of aliases.
        """
        return self.path