from __future__ import annotations
import dataclasses
import enum
import typing
@dataclasses.dataclass(frozen=True)
class _BoxSymbols:
    """Box symbols for drawing."""
    HORIZONTAL: str
    VERTICAL: str
    TOP_LEFT: str
    TOP_RIGHT: str
    BOTTOM_LEFT: str
    BOTTOM_RIGHT: str
    LEFT_T: str
    RIGHT_T: str
    TOP_T: str
    BOTTOM_T: str
    CROSS: str