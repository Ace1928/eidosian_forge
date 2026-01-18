from __future__ import annotations
import dataclasses
import enum
import typing
class _SHADE_SYMBOLS(typing.NamedTuple):
    """Standard shade symbols excluding empty space."""
    FULL_BLOCK: str = '█'
    DARK_SHADE: str = '▓'
    MEDIUM_SHADE: str = '▒'
    LITE_SHADE: str = '░'