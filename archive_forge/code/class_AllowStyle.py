import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class AllowStyle(Enum):
    """Values for ``cmd2.ansi.allow_style``"""
    ALWAYS = 'Always'
    NEVER = 'Never'
    TERMINAL = 'Terminal'

    def __str__(self) -> str:
        """Return value instead of enum name for printing in cmd2's set command"""
        return str(self.value)

    def __repr__(self) -> str:
        """Return quoted value instead of enum description for printing in cmd2's set command"""
        return repr(self.value)