import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class AnsiSequence:
    """Base class to create ANSI sequence strings"""

    def __add__(self, other: Any) -> str:
        """
        Support building an ANSI sequence string when self is the left operand
        e.g. Fg.LIGHT_MAGENTA + "hello"
        """
        return str(self) + str(other)

    def __radd__(self, other: Any) -> str:
        """
        Support building an ANSI sequence string when self is the right operand
        e.g. "hello" + Fg.RESET
        """
        return str(other) + str(self)