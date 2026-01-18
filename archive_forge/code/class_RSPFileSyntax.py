from __future__ import annotations
import enum
import typing as T
@enum.unique
class RSPFileSyntax(enum.Enum):
    """Which RSP file syntax the compiler supports."""
    MSVC = enum.auto()
    GCC = enum.auto()