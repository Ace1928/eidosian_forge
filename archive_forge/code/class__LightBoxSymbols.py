from __future__ import annotations
import dataclasses
import enum
import typing
@dataclasses.dataclass(frozen=True)
class _LightBoxSymbols(_BoxSymbolsWithDashes):
    """Box symbols for drawing.

    The Thin version includes extra symbols.
    Symbols are ordered as in Unicode except dashes.
    """
    TOP_LEFT_ROUNDED: str
    TOP_RIGHT_ROUNDED: str
    BOTTOM_LEFT_ROUNDED: str
    BOTTOM_RIGHT_ROUNDED: str