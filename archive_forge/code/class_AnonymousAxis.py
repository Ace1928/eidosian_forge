from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
class AnonymousAxis:
    """Used by `ParsedExpression` to represent an axis with a size (> 1), but no associated identifier.

    Note: Different instances of this class are not equal to each other, even if they have the same value.
    """

    def __init__(self, value: str) -> None:
        self.value = int(value)
        if self.value < 1:
            raise ValueError(f'Anonymous axis should have positive length, not {self.value}')

    def __repr__(self) -> str:
        return f'{self.value}-axis'