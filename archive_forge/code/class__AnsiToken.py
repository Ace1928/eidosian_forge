import re
import sys
from contextlib import suppress
from typing import Iterable, NamedTuple, Optional
from .color import Color
from .style import Style
from .text import Text
class _AnsiToken(NamedTuple):
    """Result of ansi tokenized string."""
    plain: str = ''
    sgr: Optional[str] = ''
    osc: Optional[str] = ''